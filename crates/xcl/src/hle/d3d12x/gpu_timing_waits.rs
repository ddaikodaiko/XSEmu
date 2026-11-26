use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{LazyLock, Mutex};
use std::thread::{self, ThreadId};

use super::depth_lineage_queue::{self as lineage, FenceKind};

const LATE_EPOCHS: u64 = 8;
const MAX_QUEUES: usize = 128;
const MAX_FENCES: usize = 4096;
const MAX_WAITS: usize = 4096;
const MAX_SIGNALS: usize = 8192;
const MAX_SUBMITS: usize = 8192;
const MAX_REPORT_ROWS: usize = 128;

struct Ring<K, V, const LIMIT: usize> {
    order: VecDeque<K>,
    values: HashMap<K, V>,
}

impl<K, V, const LIMIT: usize> Default for Ring<K, V, LIMIT> {
    fn default() -> Self {
        Self {
            order: VecDeque::new(),
            values: HashMap::new(),
        }
    }
}

impl<K: Copy + Eq + Hash, V, const LIMIT: usize> Ring<K, V, LIMIT> {
    fn insert_new(&mut self, key: K, value: V) {
        if self.order.len() == LIMIT {
            if let Some(old) = self.order.pop_front() {
                self.values.remove(&old);
            }
        }
        self.order.push_back(key);
        self.values.insert(key, value);
    }
}

#[derive(Default)]
struct Queue {
    epoch: u64,
    last: Option<u64>,
    active: usize,
    waits_since_submit: u32,
}

struct Submit {
    queue: usize,
    thread: ThreadId,
    next: Option<u64>,
    waits_before: u32,
    returned: bool,
    serial: bool,
    interval: Option<Interval>,
}

#[derive(Clone, Copy)]
struct Interval {
    hz: u64,
    begin: u64,
    end: u64,
}

struct Fence {
    generation: u64,
    highest: Option<u64>,
    rewound: bool,
}

struct Signal {
    queue: Option<usize>,
    observed: u64,
    ambiguous: bool,
}

struct Wait {
    queue: usize,
    epoch: u64,
    observed: u64,
    generation: Option<u64>,
    value: u64,
    thread: ThreadId,
    before: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Producer {
    Unresolved,
    Cpu,
    SameQueue,
    Queue(usize),
}

#[derive(Default)]
struct Waits {
    count: usize,
    later_signal: usize,
    with_upper: usize,
    max_upper_us: Option<f64>,
}

type Reports = BTreeMap<(u64, usize, Producer), Waits>;

#[derive(Default)]
struct Ledger {
    epoch: u64,
    observed: u64,
    queues: HashMap<usize, Queue>,
    fences: HashMap<usize, Fence>,
    signals: Ring<(u64, u64), Signal, MAX_SIGNALS>,
    submits: Ring<u64, Submit, MAX_SUBMITS>,
    waits: VecDeque<Wait>,
    lost_waits: usize,
    untracked_fence_events: usize,
}

impl Ledger {
    fn event(&mut self) -> Option<u64> {
        self.observed = self.observed.checked_add(1)?;
        Some(self.observed)
    }

    fn submitted(&mut self, queue: usize, epoch: u64, thread: ThreadId) -> u64 {
        if queue == 0 || (self.queues.len() == MAX_QUEUES && !self.queues.contains_key(&queue)) {
            return 0;
        }
        let Some(seq) = self.event() else { return 0 };
        let cursor = self.queues.entry(queue).or_default();
        cursor.epoch = cursor.epoch.max(epoch);
        let serial = cursor.active == 0;
        cursor.active += 1;
        if let Some(previous) = cursor
            .last
            .replace(seq)
            .and_then(|seq| self.submits.values.get_mut(&seq))
        {
            previous.next = Some(seq);
            previous.serial &= serial;
        }
        self.submits.insert_new(
            seq,
            Submit {
                queue,
                thread,
                next: None,
                waits_before: std::mem::take(&mut cursor.waits_since_submit),
                returned: false,
                serial,
                interval: None,
            },
        );
        seq
    }

    fn returned(&mut self, queue: usize, seq: u64) {
        if seq == 0 {
            return;
        }
        if let Some(s) = self.submits.values.get_mut(&seq) {
            if s.returned || s.queue != queue {
                return;
            }
            s.returned = true;
        }
        if let Some(q) = self.queues.get_mut(&queue) {
            q.active = q.active.saturating_sub(1);
        }
    }

    fn completed(&mut self, queue: usize, seq: u64, hz: u64, interval: Option<(u64, u64)>) {
        let Some(s) = self
            .submits
            .values
            .get_mut(&seq)
            .filter(|s| s.queue == queue)
        else {
            return;
        };
        s.interval = interval.and_then(|(begin, end)| {
            (hz != 0 && begin != 0 && end >= begin).then_some(Interval { hz, begin, end })
        });
    }

    fn fence(&mut self, fence: usize, observed: u64) -> Option<&mut Fence> {
        if fence == 0 || (self.fences.len() == MAX_FENCES && !self.fences.contains_key(&fence)) {
            self.untracked_fence_events = self.untracked_fence_events.saturating_add(1);
            return None;
        }
        Some(self.fences.entry(fence).or_insert(Fence {
            generation: observed,
            highest: None,
            rewound: false,
        }))
    }

    fn forget_fence_if(&mut self, fence: usize, cookie: u64) {
        if self
            .fences
            .get(&fence)
            .is_some_and(|f| f.generation == cookie)
        {
            self.fences.remove(&fence);
        }
    }

    fn signal(&mut self, queue: Option<usize>, fence: usize, value: u64) {
        let Some(observed) = self.event() else { return };
        let Some(fence) = self.fence(fence, observed) else {
            return;
        };
        let generation = fence.generation;
        let key = (generation, value);
        let rewound = !fence.rewound && fence.highest.is_some_and(|last| value < last);
        fence.rewound |= rewound;
        let ambiguous = fence.rewound || fence.highest == Some(value);
        fence.highest = Some(fence.highest.map_or(value, |last| last.max(value)));
        if rewound {
            for (&(seen, _), signal) in &mut self.signals.values {
                if seen == generation {
                    signal.ambiguous = true;
                }
            }
        }
        if let Some(signal) = self.signals.values.get_mut(&key) {
            signal.ambiguous = true;
        } else {
            self.signals.insert_new(
                key,
                Signal {
                    queue,
                    observed,
                    ambiguous,
                },
            );
        }
    }

    fn wait(&mut self, queue: usize, epoch: u64, fence: usize, value: u64, thread: ThreadId) {
        if queue == 0 {
            return;
        }
        let Some(observed) = self.event() else { return };
        let generation = self.fence(fence, observed).map(|f| f.generation);
        let before = self.queues.get_mut(&queue).and_then(|q| {
            q.epoch = q.epoch.max(epoch);
            q.waits_since_submit = q.waits_since_submit.saturating_add(1);
            (q.active == 0).then_some(q.last).flatten()
        });
        if self.waits.len() == MAX_WAITS {
            self.waits.pop_front();
            self.lost_waits = self.lost_waits.saturating_add(1);
        }
        self.waits.push_back(Wait {
            queue,
            epoch,
            observed,
            generation,
            value,
            thread,
            before,
        });
    }

    fn producer(&self, wait: &Wait) -> (Producer, bool) {
        let signal = wait
            .generation
            .and_then(|generation| self.signals.values.get(&(generation, wait.value)))
            .filter(|s| !s.ambiguous);
        let Some(signal) = signal else {
            return (Producer::Unresolved, false);
        };
        let producer = match signal.queue {
            None => Producer::Cpu,
            Some(0) => return (Producer::Unresolved, false),
            Some(queue) if queue == wait.queue => Producer::SameQueue,
            Some(queue) => Producer::Queue(queue),
        };
        (producer, signal.observed > wait.observed)
    }

    fn upper_us(&self, wait: &Wait) -> Option<f64> {
        let before = self.submits.values.get(&wait.before?)?;
        let after = self.submits.values.get(&before.next?)?;
        let ordered_on_consumer =
            |s: &Submit| s.queue == wait.queue && s.thread == wait.thread && s.serial && s.returned;
        if !ordered_on_consumer(before) || !ordered_on_consumer(after) || after.waits_before != 1 {
            return None;
        }
        let (before, after) = (before.interval?, after.interval?);
        if before.hz != after.hz {
            return None;
        }
        Some(after.begin.checked_sub(before.end)? as f64 * 1_000_000. / before.hz as f64)
    }

    fn boundary(&mut self, epoch: u64) -> (Reports, usize) {
        self.epoch = self.epoch.max(epoch);
        let cutoff = self.epoch.saturating_sub(LATE_EPOCHS);
        let mut reports = Reports::new();
        for _ in 0..self.waits.len() {
            let wait = self.waits.pop_front().unwrap();
            if wait.epoch >= cutoff {
                self.waits.push_back(wait);
                continue;
            }
            let (producer, later) = self.producer(&wait);
            let entry = reports
                .entry((wait.epoch, wait.queue, producer))
                .or_default();
            entry.count += 1;
            entry.later_signal += usize::from(later);
            if let Some(upper) = self.upper_us(&wait) {
                entry.with_upper += 1;
                entry.max_upper_us =
                    Some(entry.max_upper_us.map_or(upper, |prior| prior.max(upper)));
            }
        }
        self.queues
            .retain(|_, q| q.active != 0 || q.epoch >= cutoff);
        (reports, std::mem::take(&mut self.lost_waits))
    }
}

static LEDGER: LazyLock<Mutex<Ledger>> = LazyLock::new(|| Mutex::new(Ledger::default()));

pub(super) fn submitted(queue: usize, epoch: u64) -> u64 {
    LEDGER
        .lock()
        .unwrap()
        .submitted(queue, epoch, thread::current().id())
}

pub(super) fn returned(queue: usize, seq: u64) {
    LEDGER.lock().unwrap().returned(queue, seq);
}

pub(super) fn completed(queue: usize, seq: u64, hz: u64, interval: Option<(u64, u64)>) {
    LEDGER.lock().unwrap().completed(queue, seq, hz, interval);
}

pub(super) fn signal(queue: Option<usize>, _epoch: u64, fence: usize, value: u64) {
    let track = lineage::enabled();
    let edge = {
        let mut ledger = LEDGER.lock().unwrap();
        ledger.signal(queue, fence, value);
        track.then(|| {
            let f = ledger.fences.get(&fence)?;
            let signal = ledger.signals.values.get(&(f.generation, value))?;
            Some((f.generation, f.rewound, signal.ambiguous))
        }).flatten()
    };
    if let Some((generation, rewound, ambiguous)) = edge {
        lineage::observe_fence(FenceKind::Signal { rewound, ambiguous }, queue, fence, generation, value);
    }
}

pub(super) fn wait(queue: usize, epoch: u64, fence: usize, value: u64) {
    let track = lineage::enabled() && queue != 0;
    let generation = {
        let mut ledger = LEDGER.lock().unwrap();
        ledger.wait(queue, epoch, fence, value, thread::current().id());
        track.then(|| ledger.fences.get(&fence).map(|f| f.generation)).flatten()
    };
    if let Some(generation) = generation {
        lineage::observe_fence(FenceKind::Wait, Some(queue), fence, generation, value);
    }
}

pub(super) fn forget_fence(fence: usize) {
    LEDGER.lock().unwrap().fences.remove(&fence);
}

pub(super) fn fence_cookie(fence: usize) -> Option<u64> {
    LEDGER
        .lock()
        .unwrap()
        .fences
        .get(&fence)
        .map(|f| f.generation)
}

pub(super) fn forget_fence_if(fence: usize, cookie: u64) {
    LEDGER.lock().unwrap().forget_fence_if(fence, cookie);
}

pub(super) fn boundary(epoch: u64) {
    let (reports, lost, untracked) = {
        let mut ledger = LEDGER.lock().unwrap();
        let (reports, lost) = ledger.boundary(epoch);
        (
            reports,
            lost,
            std::mem::take(&mut ledger.untracked_fence_events),
        )
    };
    let omitted = reports.len().saturating_sub(MAX_REPORT_ROWS);
    for ((frame, consumer, producer), waits) in reports.into_iter().take(MAX_REPORT_ROWS) {
        let producer = match producer {
            Producer::Unresolved => "unresolved".into(),
            Producer::Cpu => "cpu".into(),
            Producer::SameQueue => "same-queue".into(),
            Producer::Queue(id) => format!("queue:{id}"),
        };
        let upper = waits
            .max_upper_us
            .map_or_else(|| "unknown".into(), |us| format!("{us:.3}"));
        crate::xlog_info!(
            "GPU timing: CPU epoch {frame} dependency consumer={consumer} observed_producer={producer} waits={} producer_signal_observed_after_wait={} gpu_wait_upper_known={} gpu_wait_unknown={} gpu_wait_upper_max_us={upper}",
            waits.count, waits.later_signal, waits.with_upper, waits.count - waits.with_upper
        );
    }
    if lost != 0 || omitted != 0 || untracked != 0 {
        crate::xlog_info!("GPU timing: CPU epoch {epoch} dependency ledger dropped_waits={lost} omitted_rows={omitted} untracked_fence_events={untracked}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONSUMER: usize = 7;
    const FENCE: usize = 13;
    const EPOCH: u64 = 1;
    const HZ: u64 = 1_000_000;

    #[test]
    fn stale_release_cookie_cannot_remove_a_reused_fence() {
        let mut ledger = Ledger::default();
        ledger.signal(Some(1), FENCE, 5);
        let old = ledger.fences[&FENCE].generation;
        ledger.forget_fence_if(FENCE, old);
        assert!(!ledger.fences.contains_key(&FENCE));
        ledger.signal(Some(2), FENCE, 5);
        let new = ledger.fences[&FENCE].generation;
        assert_ne!(old, new);
        ledger.forget_fence_if(FENCE, old);
        assert_eq!(ledger.fences[&FENCE].generation, new);
    }

    #[test]
    fn a_full_fence_ledger_counts_untracked_events_and_recovers_after_release() {
        let mut ledger = Ledger::default();
        for address in 1..=MAX_FENCES {
            ledger.signal(Some(1), address, 1);
        }
        ledger.signal(Some(1), MAX_FENCES + 1, 1);
        ledger.wait(CONSUMER, EPOCH, MAX_FENCES + 1, 1, thread::current().id());
        assert_eq!(ledger.untracked_fence_events, 2);
        let old = ledger.fences[&FENCE].generation;
        ledger.forget_fence_if(FENCE, old);
        ledger.signal(Some(2), MAX_FENCES + 1, 2);
        assert!(ledger.fences.contains_key(&(MAX_FENCES + 1)));
    }

    fn wait(ledger: &mut Ledger, value: u64) {
        ledger.wait(CONSUMER, EPOCH, FENCE, value, thread::current().id());
    }

    fn submit(ledger: &mut Ledger, interval: Option<(u64, u64)>) -> u64 {
        let seq = ledger.submitted(CONSUMER, EPOCH, thread::current().id());
        ledger.returned(CONSUMER, seq);
        ledger.completed(CONSUMER, seq, HZ, interval);
        seq
    }

    #[test]
    fn only_exact_values_on_the_observed_monotonic_fence_match() {
        let mut ledger = Ledger::default();
        ledger.signal(Some(21), FENCE, 2);
        ledger.signal(None, FENCE, 4);
        ledger.signal(Some(CONSUMER), FENCE, 5);
        ledger.signal(Some(0), FENCE, 6);
        for value in 1..=6 {
            wait(&mut ledger, value);
        }
        let found: Vec<_> = ledger
            .waits
            .iter()
            .map(|wait| ledger.producer(wait))
            .collect();
        assert_eq!(
            found,
            [
                (Producer::Unresolved, false),
                (Producer::Queue(21), false),
                (Producer::Unresolved, false),
                (Producer::Cpu, false),
                (Producer::SameQueue, false),
                (Producer::Unresolved, false),
            ]
        );
    }

    #[test]
    fn later_matching_signal_is_distinct_and_equal_signals_only_poison_their_value() {
        let mut ledger = Ledger::default();
        ledger.signal(Some(10), FENCE, 6);
        wait(&mut ledger, 6);
        wait(&mut ledger, 8);
        ledger.signal(Some(11), FENCE, 8);
        assert_eq!(
            ledger.producer(&ledger.waits[1]),
            (Producer::Queue(11), true)
        );
        ledger.signal(Some(12), FENCE, 8);
        assert_eq!(
            ledger.producer(&ledger.waits[0]),
            (Producer::Queue(10), false)
        );
        assert_eq!(
            ledger.producer(&ledger.waits[1]),
            (Producer::Unresolved, false)
        );
    }

    #[test]
    fn strict_rewind_poisons_prior_and_future_matches_for_only_that_generation() {
        let mut ledger = Ledger::default();
        ledger.signal(Some(10), FENCE, 4);
        ledger.signal(Some(11), FENCE, 8);
        wait(&mut ledger, 4);
        wait(&mut ledger, 8);
        assert_eq!(
            ledger.producer(&ledger.waits[0]),
            (Producer::Queue(10), false)
        );
        ledger.signal(Some(12), FENCE, 6);
        ledger.signal(Some(13), FENCE, 9);
        wait(&mut ledger, 6);
        wait(&mut ledger, 9);
        assert!(ledger
            .waits
            .iter()
            .all(|wait| ledger.producer(wait) == (Producer::Unresolved, false)));
        ledger.fences.remove(&FENCE);
        ledger.signal(Some(14), FENCE, 9);
        wait(&mut ledger, 9);
        assert_eq!(
            ledger.producer(&ledger.waits[3]),
            (Producer::Unresolved, false)
        );
        assert_eq!(
            ledger.producer(&ledger.waits[4]),
            (Producer::Queue(14), false)
        );
    }

    #[test]
    fn deleting_a_fence_does_not_pair_an_old_wait_with_the_reused_address() {
        let mut ledger = Ledger::default();
        ledger.signal(Some(11), FENCE, 4);
        wait(&mut ledger, 4);
        wait(&mut ledger, 7);
        ledger.fences.remove(&FENCE);
        ledger.signal(Some(12), FENCE, 7);
        wait(&mut ledger, 7);
        assert_eq!(
            ledger.producer(&ledger.waits[0]),
            (Producer::Queue(11), false)
        );
        assert_eq!(
            ledger.producer(&ledger.waits[1]),
            (Producer::Unresolved, false)
        );
        assert_eq!(
            ledger.producer(&ledger.waits[2]),
            (Producer::Queue(12), false)
        );
        assert_ne!(ledger.waits[1].generation, ledger.waits[2].generation);
    }

    #[test]
    fn a_single_wait_between_ordered_complete_submissions_has_an_upper_bound() {
        let mut ledger = Ledger::default();
        let before = submit(&mut ledger, None);
        wait(&mut ledger, 9);
        submit(&mut ledger, Some((50, 80)));
        assert_eq!(ledger.upper_us(&ledger.waits[0]), None);
        ledger.completed(CONSUMER, before, HZ, Some((10, 30)));
        assert_eq!(ledger.upper_us(&ledger.waits[0]), Some(20.));
        let (recent, _) = ledger.boundary(EPOCH + LATE_EPOCHS);
        assert!(recent.is_empty());
        let (retired, _) = ledger.boundary(EPOCH + LATE_EPOCHS + 1);
        let row = &retired[&(EPOCH, CONSUMER, Producer::Unresolved)];
        assert_eq!(
            (row.count, row.with_upper, row.max_upper_us),
            (1, 1, Some(20.))
        );
    }

    #[test]
    fn missing_scopes_gaps_and_multiple_waits_do_not_borrow_a_later_interval() {
        let mut ledger = Ledger::default();
        submit(&mut ledger, Some((10, 20)));
        wait(&mut ledger, 9);
        submit(&mut ledger, None);
        submit(&mut ledger, Some((90, 100)));
        assert_eq!(ledger.upper_us(&ledger.waits[0]), None);
        wait(&mut ledger, 10);
        wait(&mut ledger, 11);
        submit(&mut ledger, Some((120, 130)));
        assert!(ledger
            .waits
            .iter()
            .all(|wait| ledger.upper_us(wait).is_none()));
    }

    #[test]
    fn overlapping_callbacks_other_threads_and_nonmonotonic_scopes_stay_unknown() {
        let mut ledger = Ledger::default();
        let thread = thread::current().id();
        let first = ledger.submitted(CONSUMER, EPOCH, thread);
        wait(&mut ledger, 7);
        let second = ledger.submitted(CONSUMER, EPOCH, thread);
        ledger.returned(CONSUMER, second);
        ledger.returned(CONSUMER, first);
        ledger.completed(CONSUMER, first, HZ, Some((10, 20)));
        ledger.completed(CONSUMER, second, HZ, Some((25, 30)));
        assert_eq!(ledger.upper_us(&ledger.waits[0]), None);

        submit(&mut ledger, Some((50, 70)));
        wait(&mut ledger, 8);
        submit(&mut ledger, Some((60, 80)));
        assert_eq!(ledger.upper_us(&ledger.waits[1]), None);

        wait(&mut ledger, 9);
        let other = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        let following = ledger.submitted(CONSUMER, EPOCH, other);
        ledger.returned(CONSUMER, following);
        ledger.completed(CONSUMER, following, HZ, Some((100, 120)));
        assert_eq!(ledger.upper_us(&ledger.waits[2]), None);
    }
}
