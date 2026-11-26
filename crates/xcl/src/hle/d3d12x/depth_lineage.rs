use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::depth_lineage_queue as queue;
use super::depth_state::Range;

const OWNERS: usize = 1024;
const EVENTS: usize = 96;
const PINNED_RANGES: usize = 128;
const GLOBAL_RECORDINGS: usize = 512;
static SEQUENCE: AtomicU64 = AtomicU64::new(1);
static DROPPED: AtomicUsize = AtomicUsize::new(0);
static REPORTS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct Event {
    seq: u64,
    epoch: u64,
    resource: usize,
    source: usize,
    list: usize,
    generation: u64,
    range: Range,
    kind: &'static str,
    detail: String,
}

#[derive(Clone)]
pub(super) struct PendingPin(Weak<Mutex<History>>, Arc<Event>);

impl PendingPin {
    pub(super) fn same_slot(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0) && self.1.pin_key() == other.1.pin_key()
    }

    pub(super) fn submitted(self, submit: queue::SubmitPoint) {
        let Some(history) = self.0.upgrade() else { return };
        let mut h = history.lock().unwrap();
        if let Some(pin) = h.pins.iter_mut().find(|p| p.cpu.pin_key() == self.1.pin_key()) {
            pin.submitted(self.1, submit);
        }
    }
}

impl Event {
    fn pin_key(&self) -> (usize, Range, &'static str) {
        (self.resource, self.range, self.kind)
    }

    fn pin(&self) -> bool {
        self.kind.starts_with("dsv_") || self.kind.starts_with("slot146-guest-pre-twin") || matches!(self.kind,
            "resource-created" | "first-view" | "twin-created" | "twin-uav-view" | "null-depth-uav-view"
            | "guest-legacy-named" | "guest-legacy-transition" | "guest-legacy-uav-egress"
            | "guest-legacy-split-begin" | "guest-legacy-split-end" | "guest-alias-barrier"
            | "guest-discard" | "slot146-guest-texture" | "slot146-guest-uav-egress"
            | "slot146-guest-split" | "slot146-guest-discard" | "slot146-guest-activation"
            | "native-legacy-transition" | "native-legacy-split-begin" | "native-legacy-split-end"
            | "native-texture-transition" | "native-texture-split-begin" | "native-texture-split-end"
            | "native-texture-activation" | "native-texture-discard" | "native-alias-barrier"
            | "native-alias-activate" | "native-discard"
            | "typed_uav_shadow_binding" | "typed_uav_null_binding" | "typed_uav_null_stencil_binding"
            | "copy-in-recorded" | "copy-out-recorded" | "copy-in-skipped" | "copy-out-skipped")
    }
}

#[derive(Clone)]
struct Pin {
    cpu: Arc<Event>,
    first_submit: Option<(Arc<Event>, queue::SubmitPoint)>,
    unordered_with: Option<queue::SubmitPoint>,
}

impl Pin {
    fn submitted(&mut self, event: Arc<Event>, point: queue::SubmitPoint) {
        if let Some((first, prior)) = &self.first_submit {
            if point.unordered_with(*prior) {
                self.unordered_with = Some(if point.key() < prior.key() { *prior } else { point });
            }
            if (point.key(), event.seq) >= (prior.key(), first.seq) {
                return;
            }
        }
        self.first_submit = Some((event, point));
    }
}

#[derive(Default)]
struct History {
    events: VecDeque<Event>,
    replaced: usize,
    evicted: usize,
    last_report: Option<u64>,
    pins: Vec<Pin>,
    pin_drops: usize,
}

impl History {
    fn push(&mut self, e: Event, cap: usize) {
        let mut same = self.events.iter().enumerate().filter(|(_, v)| {
            (v.resource, v.range, v.kind, v.source) == (e.resource, e.range, e.kind, e.source)
        });
        let first = same.next().map(|(i, _)| i);
        if let Some(i) = first.filter(|_| same.next().is_some()) {
            self.events.remove(i);
            self.replaced += 1;
        }
        if self.events.len() == cap {
            let index = self.events.iter().position(|v| !matches!(v.kind, "first-view" | "resource-created" | "twin-created")).unwrap_or(0);
            self.events.remove(index);
            self.evicted += 1;
        }
        self.events.push_back(e);
    }
}

type Log = Arc<Mutex<History>>;
#[derive(Default)]
struct Registry {
    owners: HashMap<usize, (usize, Log)>,
    twins: HashMap<usize, usize>,
    globals: HashMap<(usize, u64), Log>,
    order: VecDeque<(usize, u64)>,
}
static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));

pub(super) fn twin_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("XCL_DEPTH_TWIN_EGRESS").as_deref() == Ok("1"))
}

pub(super) fn enabled() -> bool {
    queue::enabled() && super::depth_capture_views::enabled()
}

pub(super) fn watch(owner: usize) {
    if owner == 0 || !enabled() { return }
    let mut r = REGISTRY.lock().unwrap();
    if r.owners.contains_key(&owner) { return }
    if r.owners.len() >= OWNERS {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    r.owners.insert(owner, (0, Log::default()));
}

pub(super) fn start(owner: usize, twin: usize) {
    if !enabled() || owner == 0 || twin == 0 { return }
    watch(owner);
    let mut r = REGISTRY.lock().unwrap();
    if let Some((slot, _)) = r.owners.get_mut(&owner) {
        let old = std::mem::replace(slot, twin);
        r.twins.remove(&old);
        r.twins.insert(twin, owner);
    }
}

pub(super) fn forget(resource: usize) {
    if !enabled() { return }
    let mut r = REGISTRY.lock().unwrap();
    if let Some((twin, _)) = r.owners.remove(&resource) {
        r.twins.remove(&twin);
    }
    if let Some(owner) = r.twins.remove(&resource) {
        if let Some((slot, _)) = r.owners.get_mut(&owner) {
            *slot = 0;
        }
    }
}

pub(super) fn interested(resource: usize) -> bool {
    if resource == 0 || !enabled() { return false }
    let r = REGISTRY.lock().unwrap();
    r.owners.contains_key(&resource) || r.twins.contains_key(&resource)
}

pub(super) fn note(resource: usize, list: usize, range: Range, kind: &'static str, detail: impl FnOnce() -> String) {
    note_key(resource, list, range, kind, 0, detail)
}

pub(super) fn note_key(resource: usize, list: usize, range: Range, kind: &'static str, source: usize, detail: impl FnOnce() -> String) {
    if !enabled() { return }
    let generation = if list != 0 { queue::recording(list) } else { 0 };
    let log = {
        let mut r = REGISTRY.lock().unwrap();
        if resource == 0 {
            if list == 0 || generation == 0 { return }
            let key = (list, generation);
            if !r.globals.contains_key(&key) {
                if r.order.len() == GLOBAL_RECORDINGS {
                    if let Some(old) = r.order.pop_front() { r.globals.remove(&old); }
                }
                r.order.push_back(key);
            }
            r.globals.entry(key).or_default().clone()
        } else {
            let owner = r.twins.get(&resource).copied().unwrap_or(resource);
            let Some((_, log)) = r.owners.get(&owner) else { return };
            log.clone()
        }
    };
    let detail = detail();
    let epoch = super::present::counters().1;
    let mut history = log.lock().unwrap();
    let e = Event { seq: SEQUENCE.fetch_add(1, Ordering::Relaxed), epoch, resource, source, list, generation, range, kind, detail };
    let pending = if resource != 0 && twin_enabled() && e.pin() {
        let e = Arc::new(e.clone());
        if history.pins.iter().any(|p| p.cpu.pin_key() == e.pin_key()) {
            Some(e)
        } else if history.pins.len() < PINNED_RANGES {
            history.pins.push(Pin { cpu: e.clone(), first_submit: None, unordered_with: None });
            Some(e)
        } else {
            history.pin_drops += 1;
            None
        }
    } else {
        None
    };
    history.push(e, if resource == 0 { 12 } else { EVENTS });
    drop(history);
    if let Some(e) = pending.filter(|e| e.list != 0) {
        queue::pin_event(list, generation, PendingPin(Arc::downgrade(&log), e));
    }
}

pub(super) fn report_writer_input(owner: usize, list: usize, generation: u64) {
    let history = REGISTRY.lock().unwrap().owners.get(&owner).map(|(_, h)| h.clone());
    let Some(history) = history else {
        crate::xlog_info!("writer depth lineage: input owner={owner:#x} list={list:#x}/{generation} untracked");
        return
    };
    let (events, evicted, replaced) = {
        let h = history.lock().unwrap();
        (h.events.iter().rev().filter(|e| e.resource == owner && e.range.covers(0) &&
            (e.kind.starts_with("dsv_") || matches!(e.kind, "resource-created" | "copy-out-recorded" | "native-discard" | "guest-discard")))
            .take(24).cloned().collect::<Vec<_>>(), h.evicted, h.replaced)
    };
    crate::xlog_info!("writer depth lineage: input owner={owner:#x} list={list:#x}/{generation} recent recorded DSV/clear/draw events={} evicted={evicted} replaced={replaced}; last CPU record is not last GPU writer", events.len());
    for e in events {
        let joined = queue::describe(e.list, e.generation);
        if e.generation != 0 { queue::watch_selected(e.list, e.generation); }
        crate::xlog_info!("writer depth lineage: input owner={owner:#x} host_event={} epoch={} producer={:#x}/{} {} {} submit={joined}", e.seq, e.epoch, e.list, e.generation, e.kind, e.detail);
    }
}

pub(super) fn report(owner: usize, sub: u32, sibling: Option<u32>, list: usize) {
    if !enabled() || REPORTS.load(Ordering::Relaxed) >= 3 { return }
    let generation = queue::recording(list);
    let epoch = super::present::counters().1;
    let (twin, history, global) = {
        let r = REGISTRY.lock().unwrap();
        let (twin, log) = r.owners.get(&owner).cloned().unwrap_or_default();
        (twin, log, r.globals.get(&(list, generation)).cloned())
    };
    let (mut events, replaced, evicted, pins, pin_drops) = {
        let mut h = history.lock().unwrap();
        if h.last_report.is_some_and(|prior| epoch.saturating_sub(prior) < 4) { return }
        if REPORTS.fetch_add(1, Ordering::Relaxed) >= 3 { return }
        h.last_report = Some(epoch);
        let e: Vec<_> = h.events.iter().filter(|e| e.range.covers(sub) || sibling.is_some_and(|s| e.range.covers(s))).cloned().collect();
        let pins = h.pins.iter().filter(|p| p.cpu.range == Range::Unknown || p.cpu.range.covers(sub)
            || sibling.is_some_and(|s| p.cpu.range.covers(s))).cloned().collect::<Vec<_>>();
        (e, h.replaced, h.evicted, pins, h.pin_drops)
    };
    if let Some(g) = global {
        events.extend(g.lock().unwrap().events.iter().cloned());
    }
    events.sort_unstable_by_key(|e| e.seq);
    let no_producer = !events.iter().any(|e| e.resource != 0 && e.range.covers(sub) &&
        matches!(e.kind, "typed_uav_shadow_binding" | "dsv_clear_recorded" | "dsv_depth_write_capable_draw" | "copy-out-recorded"));
    let missing = if no_producer { "no retained producer command/binding matching the selected depth subresource" }
        else { "recorded candidates do not establish GPU-written depth codes or samples passing EQUAL" };
    queue::watch_selected(list, generation);
    crate::xlog_info!("depth lineage: selected EQUAL epoch={epoch} owner={owner:#x} twin={twin:#x} sub={sub} list={list:#x}/{generation} host_sequence={} replaced_same_kind_range={replaced} evicted={evicted} owner_admission_drops={} first missing: {missing}; host record sequence is not GPU execution order",
        SEQUENCE.load(Ordering::Relaxed), DROPPED.load(Ordering::Relaxed));
    let mut joins = vec![(list, generation)];
    if twin_enabled() {
        crate::xlog_info!("depth twin egress: pinned event drops={pin_drops} queue pin gaps={}; first submitted means the recording belonged to the earliest observed Execute entry among calls that returned, not GPU execution; a guest note does not prove native barrier emission; unhooked creation/barriers, opaque or unnamed guest barriers and unknown recording generations are gaps", queue::pin_gaps());
        for resource in [owner, twin].into_iter().filter(|&r| r != 0) {
            let arena = super::vram::arena_lineage(resource)
                .unwrap_or_else(|| "no observed arena placement (committed, tiled, slab, external or missed)".into());
            crate::xlog_info!("depth twin egress: resource={resource:#x} {arena}");
        }
        for (scope, selected_sub) in [("depth", Some(sub)), ("sibling-stencil", sibling), ("unresolved-owner-range", None)] {
            if scope == "sibling-stencil" && selected_sub.is_none() { continue }
            let mut selected: Vec<Pin> = Vec::new();
            for pin in pins.iter().filter(|p| selected_sub.map_or(p.cpu.range == Range::Unknown, |s| p.cpu.range.covers(s))
                && (scope != "sibling-stencil" || p.cpu.resource == owner))
            {
                if let Some(merged) = selected.iter_mut().find(|p| {
                    (p.cpu.resource, p.cpu.kind) == (pin.cpu.resource, pin.cpu.kind)
                }) {
                    if pin.cpu.seq < merged.cpu.seq { merged.cpu = pin.cpu.clone(); }
                    if let Some((event, point)) = &pin.first_submit {
                        merged.submitted(event.clone(), *point);
                    }
                    merged.unordered_with = merged.unordered_with.or(pin.unordered_with);
                } else {
                    selected.push(pin.clone());
                }
            }
            selected.sort_unstable_by_key(|p| p.cpu.seq);
            for pin in selected {
                let e = &pin.cpu;
                crate::xlog_info!("depth twin egress: {scope} sub={selected_sub:?} resource={:#x} {} first-cpu-recorded host#{} list={:#x}/{} range={:?}: {}",
                    e.resource, e.kind, e.seq, e.list, e.generation, e.range, e.detail);
                if e.list != 0 && !joins.contains(&(e.list, e.generation)) { joins.push((e.list, e.generation)); }
                if let Some((entry, point)) = pin.first_submit {
                    crate::xlog_info!("depth twin egress: {scope} sub={selected_sub:?} resource={:#x} {} first-host-submitted host#{} list={:#x}/{} range={:?} submit={point} unresolved-order-with={:?}: {}",
                        entry.resource, entry.kind, entry.seq, entry.list, entry.generation, entry.range, pin.unordered_with, entry.detail);
                    if !joins.contains(&(entry.list, entry.generation)) { joins.push((entry.list, entry.generation)); }
                } else if e.list != 0 {
                    crate::xlog_info!("depth twin egress: {scope} sub={selected_sub:?} resource={:#x} {} first-host-submitted=not-observed-at-report",
                        e.resource, e.kind);
                }
            }
        }
    }
    for e in events {
        if e.list != 0 && !joins.contains(&(e.list, e.generation)) { joins.push((e.list, e.generation)); }
        let scope = if e.resource == 0 { "list-global (not a texture)" }
            else if !e.range.covers(sub) { "sibling stencil (not a depth write)" }
            else if sibling.is_some_and(|s| e.range.covers(s)) { "whole resource or both planes" }
            else { "depth" };
        crate::xlog_info!("depth lineage: host#{} epoch={} resource={:#x} list={:#x}/{} range={:?} scope={scope} {}: {}",
            e.seq, e.epoch, e.resource, e.list, e.generation, e.range, e.kind, e.detail);
    }
    for (recorded_list, g) in joins {
        crate::xlog_info!("depth lineage: list={recorded_list:#x}/{g} {}", queue::describe(recorded_list, g));
    }
}
