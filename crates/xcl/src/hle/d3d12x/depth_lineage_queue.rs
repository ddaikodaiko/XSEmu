use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::thread::{self, ThreadId};

use super::depth_lineage::PendingPin;

const MAX_LIVE_LISTS: usize = 8192;
const MAX_SUBMISSIONS: usize = 4096;
const MAX_MEMBERS: usize = 256;
const MAX_FENCE_EVENTS: usize = 8192;
const MAX_WATCHES: usize = 64;
const MAX_REPORT_SUBMISSIONS: usize = 8;
const MAX_REPORT_EDGES: usize = 4;
const MAX_RECORDING_PINS: usize = 128;
const MAX_LIVE_PINS: usize = 8192;
const MAX_SUBMISSION_PINS: usize = 256;

type Recording = (usize, u64);

#[derive(Clone, Copy, Debug)]
pub(super) struct SubmitPoint {
    id: u64,
    end: u64,
    queue: usize,
    member: usize,
}

impl SubmitPoint {
    pub(super) fn key(self) -> (u64, usize) { (self.id, self.member) }

    pub(super) fn unordered_with(self, other: Self) -> bool {
        self.id != other.id && (self.queue == 0 || self.queue != other.queue
            || self.id < other.end && other.id < self.end)
    }
}

impl std::fmt::Display for SubmitPoint {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(out, "entry={} return={} q={} member={}", self.id, self.end, self.queue, self.member)
    }
}

#[derive(Clone, Copy)]
pub(super) enum FenceKind {
    Signal { rewound: bool, ambiguous: bool },
    Wait,
}

struct Edge {
    observed: u64,
    queue: Option<usize>,
    thread: ThreadId,
    fence: usize,
    generation: u64,
    value: u64,
    kind: FenceKind,
}

struct Submit {
    id: u64,
    raw_queue: usize,
    queue: usize,
    timing_seq: u64,
    thread: ThreadId,
    members: Vec<Recording>,
    member_count: usize,
    returned: Option<u64>,
    retired: Option<(usize, usize, u64)>,
    pins: Vec<(usize, PendingPin)>,
}

impl Submit {
    fn covers_with(&self, signal: &Edge) -> bool {
        self.queue != 0
            && signal.queue == Some(self.queue)
            && signal.thread == self.thread
            && self.returned.is_some_and(|end| end < signal.observed)
            && matches!(signal.kind, FenceKind::Signal { .. })
    }
}

struct Watch {
    recording: Recording,
    first_submit: Option<u64>,
    reported_signal: bool,
    reported_wait: bool,
}

#[derive(Default)]
struct Ledger {
    next_recording: usize,
    observed: u64,
    live: HashMap<usize, u64>,
    pins: HashMap<Recording, Vec<PendingPin>>,
    pin_gaps: usize,
    live_pins: usize,
    submissions: VecDeque<Submit>,
    edges: VecDeque<Edge>,
    watches: VecDeque<Watch>,
    evicted_submissions: u64,
    evicted_edges: u64,
}

impl Ledger {
    fn drop_pins(&mut self, key: Recording) {
        if let Some(events) = self.pins.remove(&key) {
            self.live_pins -= events.len();
        }
    }

    fn event(&mut self) -> Option<u64> {
        self.observed = self.observed.checked_add(1)?;
        Some(self.observed)
    }

    fn submission(&self, id: u64) -> Option<&Submit> {
        let index = self.submissions.binary_search_by_key(&id, |s| s.id).ok()?;
        self.submissions.get(index)
    }

    fn watch(&mut self, recording: Recording, first_submit: Option<u64>) {
        if self.watches.iter().any(|w| w.recording == recording) {
            return;
        }
        if self.watches.len() == MAX_WATCHES {
            self.watches.pop_front();
        }
        self.watches.push_back(Watch {
            recording,
            first_submit,
            reported_signal: false,
            reported_wait: false,
        });
    }

    fn edges_for(&self, submit: &Submit) -> (Vec<&Edge>, Vec<&Edge>) {
        if submit.queue == 0 {
            return (Vec::new(), Vec::new());
        }
        let previous = self
            .submissions
            .iter()
            .rev()
            .find(|s| s.id < submit.id && s.queue == submit.queue && s.thread == submit.thread);
        let waits = self.edges.iter().rev().filter(|e| {
            e.queue == Some(submit.queue)
                && e.thread == submit.thread
                && e.observed < submit.id
                && previous.is_none_or(|s| s.returned.is_some_and(|end| end < e.observed))
                && matches!(e.kind, FenceKind::Wait)
        });
        let signals = self.edges.iter().filter(|e| submit.covers_with(e));
        (
            waits.take(MAX_REPORT_EDGES).collect(),
            signals.take(MAX_REPORT_EDGES).collect(),
        )
    }

    fn signal_usable(&self, signal: &Edge) -> bool {
        matches!(
            signal.kind,
            FenceKind::Signal {
                ambiguous: false,
                ..
            }
        ) && !self.edges.iter().any(|e| {
            e.generation == signal.generation
                && e.observed != signal.observed
                && matches!(e.kind, FenceKind::Signal { rewound: true, .. })
        }) && !self.edges.iter().any(|e| {
            e.generation == signal.generation
                && e.observed != signal.observed
                && matches!(e.kind, FenceKind::Signal { .. })
                && (e.value == signal.value || (e.queue != signal.queue && e.value >= signal.value))
        })
    }

    fn render(&self, recording: Recording) -> String {
        let mut out = String::new();
        let mut matched = self
            .submissions
            .iter()
            .rev()
            .filter(|s| s.members.contains(&recording));
        for s in matched.by_ref().take(MAX_REPORT_SUBMISSIONS) {
            let (waits, signals) = self.edges_for(s);
            let _ = write!(
                out,
                " [submit={} q={} raw_queue={:#x} timing_seq={} members={}/{} retained_occurrences={} status={}",
                s.id,
                s.queue,
                s.raw_queue,
                s.timing_seq,
                s.members.len(),
                s.member_count,
                s.members
                    .iter()
                    .filter(|&&member| member == recording)
                    .count(),
                if s.returned.is_some() {
                    "host-returned"
                } else {
                    "host-call-entered"
                },
            );
            if let Some((flight, fence, value)) = s.retired {
                let _ = write!(
                    out,
                    " retirement=timing-marker-retired flight={flight} marker={fence:#x}:{value}"
                );
            } else {
                out.push_str(" retirement=unknown");
            }
            for wait in waits {
                let _ = write!(
                    out,
                    " observed-prior-wait={:#x}/{}:{}@{}",
                    wait.fence, wait.generation, wait.value, wait.observed
                );
                if let Some(signal) = self
                    .edges
                    .iter()
                    .find(|e| {
                        e.generation == wait.generation
                            && e.value == wait.value
                            && matches!(e.kind, FenceKind::Signal { .. })
                    })
                    .filter(|e| self.signal_usable(e))
                {
                    let _ = write!(
                        out,
                        "/exact-signal-candidate={:?}@{}",
                        signal.queue, signal.observed
                    );
                }
            }
            for signal in signals {
                let _ = write!(
                    out,
                    " observed-covering-signal={:#x}/{}:{}@{}",
                    signal.fence, signal.generation, signal.value, signal.observed
                );
                if self.signal_usable(signal) {
                    for wait in self
                        .edges
                        .iter()
                        .filter(|e| {
                            e.generation == signal.generation
                                && e.value == signal.value
                                && e.queue != signal.queue
                                && matches!(e.kind, FenceKind::Wait)
                        })
                        .take(MAX_REPORT_EDGES)
                    {
                        let _ = write!(
                            out,
                            "/exact-wait-candidate={:?}@{}",
                            wait.queue, wait.observed
                        );
                    }
                }
            }
            out.push(']');
        }
        if matched.next().is_some() {
            out.push_str(" more-submissions-omitted");
        }
        if out.is_empty() {
            out.push_str(" recorded; no retained submission");
        }
        let _ = write!(
            out,
            " evicted_submissions={} evicted_fence_observations={}",
            self.evicted_submissions, self.evicted_edges
        );
        out
    }
}

static LEDGER: LazyLock<Mutex<Ledger>> = LazyLock::new(|| Mutex::new(Ledger::default()));

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| super::depth_lineage::twin_enabled()
        || std::env::var("XCL_DEPTH_LINEAGE").is_ok_and(|v| v.trim() == "1"))
}

pub(super) fn opened(list: usize) -> Option<usize> {
    if !enabled() || list == 0 {
        return None;
    }
    let mut ledger = LEDGER.lock().unwrap();
    let id = ledger.next_recording.checked_add(1)?;
    ledger.next_recording = id;
    if ledger.live.len() < MAX_LIVE_LISTS || ledger.live.contains_key(&list) {
        if let Some(old) = ledger.live.insert(list, id as u64) {
            ledger.drop_pins((list, old));
        }
    }
    Some(id)
}

pub(super) fn forget(list: usize, generation: Option<u64>) {
    if !enabled() {
        return;
    }
    let mut ledger = LEDGER.lock().unwrap();
    if ledger
        .live
        .get(&list)
        .is_some_and(|&id| generation.is_none_or(|expected| expected == id))
    {
        if let Some(old) = ledger.live.remove(&list) {
            ledger.drop_pins((list, old));
        }
    }
}

pub(super) fn recording(list: usize) -> u64 {
    if !enabled() {
        return 0;
    }
    LEDGER.lock().unwrap().live.get(&list).copied().unwrap_or(0)
}

pub(super) fn pin_event(list: usize, generation: u64, event: PendingPin) {
    let mut ledger = LEDGER.lock().unwrap();
    if generation == 0 || ledger.live.get(&list) != Some(&generation) {
        ledger.pin_gaps += 1;
        return;
    }
    let at_cap = ledger.live_pins >= MAX_LIVE_PINS;
    let events = ledger.pins.entry((list, generation)).or_default();
    if events.iter().any(|v| v.same_slot(&event)) {
        return;
    }
    if !at_cap && events.len() < MAX_RECORDING_PINS {
        events.push(event);
        ledger.live_pins += 1;
    } else {
        ledger.pin_gaps += 1;
    }
}

pub(super) fn pin_gaps() -> usize {
    LEDGER.lock().unwrap().pin_gaps
}

pub(super) fn describe(list: usize, generation: u64) -> String {
    if !enabled() || generation == 0 {
        return "recording=unknown".into();
    }
    LEDGER.lock().unwrap().render((list, generation))
}

pub(super) fn watch_selected(list: usize, generation: u64) {
    if !enabled() || generation == 0 {
        return;
    }
    let recording = (list, generation);
    let mut ledger = LEDGER.lock().unwrap();
    let returned = ledger
        .submissions
        .iter()
        .find(|s| s.returned.is_some() && s.members.contains(&recording))
        .map(|s| s.id);
    if returned.is_some()
        || ledger.live.get(&list) == Some(&generation)
        || ledger
            .submissions
            .iter()
            .any(|s| s.members.contains(&recording))
    {
        ledger.watch(recording, returned);
    }
}

pub(super) fn enter(raw_queue: usize, queue: usize, timing_seq: u64, lists: &[*mut c_void]) -> u64 {
    if !enabled() {
        return 0;
    }
    let mut ledger = LEDGER.lock().unwrap();
    let Some(id) = ledger.event() else { return 0 };
    super::material_gate_views::writer::entered(lists, &ledger.live);
    super::material_gate_views::prepass::entered(lists, &ledger.live);
    let mut members: Vec<_> = lists
        .iter()
        .take(MAX_MEMBERS)
        .map(|&raw| {
            let list = raw as usize;
            (list, ledger.live.get(&list).copied().unwrap_or(0))
        })
        .collect();
    if lists.len() > MAX_MEMBERS && !ledger.watches.is_empty() {
        for &raw in &lists[MAX_MEMBERS..] {
            let list = raw as usize;
            let Some(&generation) = ledger.live.get(&list) else {
                continue;
            };
            let recording = (list, generation);
            if ledger.watches.iter().any(|w| w.recording == recording)
                && !members.contains(&recording)
            {
                members.push(recording);
            }
        }
    }
    let mut pins = Vec::new();
    if !ledger.pins.is_empty() {
        let mut omitted = 0;
        for (member, &raw) in lists.iter().enumerate() {
            let list = raw as usize;
            let Some(&generation) = ledger.live.get(&list) else { continue };
            let Some(events) = ledger.pins.get(&(list, generation)) else { continue };
            let count = events.len().min(MAX_SUBMISSION_PINS - pins.len());
            pins.extend(events.iter().take(count).cloned().map(|event| (member, event)));
            omitted += events.len() - count;
        }
        ledger.pin_gaps += omitted;
    }
    if ledger.submissions.len() == MAX_SUBMISSIONS {
        if let Some(evicted) = ledger.submissions.pop_front() {
            ledger.pin_gaps += evicted.pins.len();
        }
        ledger.evicted_submissions = ledger.evicted_submissions.saturating_add(1);
    }
    ledger.submissions.push_back(Submit {
        id,
        raw_queue,
        queue,
        timing_seq,
        thread: thread::current().id(),
        members,
        member_count: lists.len(),
        returned: None,
        retired: None,
        pins,
    });
    id
}

pub(super) fn returned(id: u64) {
    if id == 0 {
        return;
    }
    let (reports, pins, end, queue) = {
        let mut ledger = LEDGER.lock().unwrap();
        let Some(end) = ledger.event() else { return };
        let ledger = &mut *ledger;
        let Some(s) = ledger
            .submissions
            .iter_mut()
            .rev()
            .find(|s| s.id == id && s.returned.is_none())
        else {
            return;
        };
        s.returned = Some(end);
        let pins = std::mem::take(&mut s.pins);
        let queue = s.queue;
        let members = &s.members;
        let reports: Vec<_> = ledger
            .watches
            .iter()
            .filter(|w| w.first_submit.is_none() && members.contains(&w.recording))
            .map(|w| w.recording)
            .collect();
        for watch in ledger
            .watches
            .iter_mut()
            .filter(|w| reports.contains(&w.recording))
        {
            watch.first_submit = Some(id);
        }
        let reports = reports.into_iter().map(|r| (r, ledger.render(r))).collect::<Vec<_>>();
        (reports, pins, end, queue)
    };
    for (member, event) in pins {
        event.submitted(SubmitPoint { id, end, queue, member });
    }
    for ((list, generation), report) in reports {
        crate::xlog_info!(
            "depth lineage queue: selected list={list:#x} recording={generation} actual-host-submit{report}"
        );
    }
}

pub(super) fn retired(id: u64, flight: usize, fence: usize, value: u64) {
    if id == 0 {
        return;
    }
    let reports = {
        let mut ledger = LEDGER.lock().unwrap();
        let Some(submit) = ledger
            .submissions
            .iter_mut()
            .rev()
            .find(|s| s.id == id && s.retired.is_none())
        else {
            return;
        };
        submit.retired = Some((flight, fence, value));
        if submit.returned.is_some() && super::material_gate_views::prepass::pending() {
            for &(list, generation) in &submit.members {
                super::material_gate_views::prepass::retired(list, generation, submit.queue, submit.timing_seq, id);
            }
        }
        if submit.returned.is_some() && super::material_gate_views::writer::pending() {
            for &(list, generation) in &submit.members {
                super::material_gate_views::writer::retired(list, generation, submit.queue, submit.timing_seq, id);
            }
        }
        ledger
            .watches
            .iter()
            .filter(|w| w.first_submit == Some(id))
            .map(|w| (w.recording, ledger.render(w.recording)))
            .collect::<Vec<_>>()
    };
    for ((list, generation), report) in reports {
        crate::xlog_info!(
            "depth lineage queue: selected list={list:#x} recording={generation} timing-marker-retirement{report}"
        );
    }
}

pub(super) fn observe_fence(
    kind: FenceKind,
    queue: Option<usize>,
    fence: usize,
    generation: u64,
    value: u64,
) {
    let reports = {
        let mut ledger = LEDGER.lock().unwrap();
        let Some(observed) = ledger.event() else {
            return;
        };
        if ledger.edges.len() == MAX_FENCE_EVENTS {
            ledger.edges.pop_front();
            ledger.evicted_edges = ledger.evicted_edges.saturating_add(1);
        }
        ledger.edges.push_back(Edge {
            observed,
            queue,
            thread: thread::current().id(),
            fence,
            generation,
            value,
            kind,
        });
        if ledger.watches.is_empty() {
            return;
        }
        let edge = ledger.edges.back().unwrap();
        let signal = match kind {
            FenceKind::Signal { .. } => Some(edge),
            FenceKind::Wait => ledger
                .edges
                .iter()
                .find(|e| {
                    e.generation == generation
                        && e.value == value
                        && e.queue != queue
                        && matches!(e.kind, FenceKind::Signal { .. })
                })
                .filter(|e| ledger.signal_usable(e)),
        };
        let Some(signal) = signal else { return };
        let reports = ledger
            .watches
            .iter()
            .filter_map(|w| {
                let already_reported = match kind {
                    FenceKind::Signal { .. } => w.reported_signal,
                    FenceKind::Wait => w.reported_wait,
                };
                (!already_reported && ledger.submission(w.first_submit?)?.covers_with(signal))
                    .then_some(w.recording)
            })
            .collect::<Vec<_>>();
        for watch in ledger
            .watches
            .iter_mut()
            .filter(|w| reports.contains(&w.recording))
        {
            match kind {
                FenceKind::Signal { .. } => watch.reported_signal = true,
                FenceKind::Wait => watch.reported_wait = true,
            }
        }
        reports
            .into_iter()
            .map(|r| (r, ledger.render(r)))
            .collect::<Vec<_>>()
    };
    for ((list, generation), report) in reports {
        crate::xlog_info!(
            "depth lineage queue: selected list={list:#x} recording={generation} fence-observation{report}"
        );
    }
}
