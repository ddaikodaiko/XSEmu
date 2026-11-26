use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::{ID3D12Resource, D3D12_COMMAND_LIST_TYPE_DIRECT};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
use super::{depth_lineage_queue, depth_state::Range, depth_uav::Shadow, material_gate_capture as gate_probe};

struct Target {
    owner: usize,
    shadow: Weak<Shadow>,
    twin: usize,
    mip: u32,
    slice: u32,
    sub: u32,
    producers: Vec<Producer>,
}

struct Producer {
    list: usize,
    generation: u64,
    owner: usize,
    twin: usize,
    sub: u32,
    shadow: Weak<Shadow>,
    index: u32,
    groups: [u32; 3],
    ambiguous: bool,
    gate: Option<gate_probe::Lease>,
}

impl Target {
    fn matches(&self, p: &Producer) -> bool {
        self.owner == p.owner && self.twin == p.twin && self.sub == p.sub
            && self.shadow.ptr_eq(&p.shadow) && self.shadow.strong_count() != 0
    }
}

static TARGET: Mutex<Option<Target>> = Mutex::new(None);
thread_local! { static BIND: RefCell<Option<Producer>> = const { RefCell::new(None) }; }

fn shader() -> Option<u64> {
    static HASH: OnceLock<Option<u64>> = OnceLock::new();
    *HASH.get_or_init(|| {
        if std::env::var("XCL_DEPTH_TWIN_EGRESS").as_deref() != Ok("1") || !super::depth_lineage::enabled() { return None }
        let s = std::env::var("XCL_DEPTH_TWIN_CAPTURE_CS").ok()?;
        u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok().filter(|n| *n != 0)
    })
}

pub(super) fn select(owner: usize, shadow: &Arc<Shadow>, twin: usize, mip: u32, slice: u32, sub: u32) {
    if shader().is_none() || !super::depth_capture::twin_due() { return }
    let mut g = TARGET.lock().unwrap();
    if g.as_ref().is_some_and(|t| t.shadow.strong_count() != 0) { return }
    *g = Some(Target { owner, shadow: Arc::downgrade(shadow), twin, mip, slice, sub, producers: Vec::new() });
    crate::xlog_info!("twin egress capture: waiting for selected producer on subsequent use; owner={owner:#x} twin={twin:#x} sub={sub}");
}

pub(super) fn begin() {
    if shader().is_some() { BIND.replace(None); }
}

pub(super) fn binding(list: usize, owner: usize, twin: Option<usize>, format: i32, index: u32, range: Range, cs: Option<u64>) {
    if shader().is_none() || shader() != cs || format != DXGI_FORMAT_R32_FLOAT.0 || !super::depth_capture::twin_due() { return }
    let Some(twin) = twin else { return };
    let (sub, shadow) = {
        let g = TARGET.lock().unwrap();
        let Some(t) = g.as_ref().filter(|t| t.owner == owner && t.twin == twin && t.shadow.strong_count() != 0 && range.covers(t.sub)) else { return };
        (t.sub, t.shadow.clone())
    };
    let generation = depth_lineage_queue::recording(list);
    if generation != 0 {
        BIND.with_borrow_mut(|bind| {
            let ambiguous = bind.as_ref().is_some_and(|p| p.generation == generation && (p.ambiguous || p.index != index));
            *bind = Some(Producer { list, generation, owner, twin, sub, shadow, index, groups: [0; 3], ambiguous, gate: None });
        });
    }
}

pub(super) fn gate_bound(list: usize) -> bool {
    BIND.with_borrow(|b| b.as_ref().is_some_and(|p| p.list == list && !p.ambiguous))
}

pub(super) fn prepare_counters(list: usize, describe: impl FnOnce() -> (Option<u32>, String)) -> Option<u64> {
    if !gate_probe::enabled() || !super::depth_capture::twin_due() {
        return None;
    }
    BIND.with_borrow_mut(|bind| {
        let p = bind.as_mut()?;
        if p.list != list || p.generation != depth_lineage_queue::recording(list) || p.gate.is_some()
            || super::list_x::tracked_list_type(list as *mut c_void) != Some(D3D12_COMMAND_LIST_TYPE_DIRECT)
            || !super::list_x::depth_copy_unpredicated(list)
        {
            return None;
        }
        if p.ambiguous {
            gate_probe::refusal("multiple native UAV indices match the selected twin");
            return None;
        }
        let (shadow, mip) = {
            let g = TARGET.lock().unwrap();
            let t = g.as_ref().filter(|t| t.matches(p))?;
            (t.shadow.upgrade()?, t.mip)
        };
        let raw = p.twin as *mut c_void;
        let desc = unsafe { ID3D12Resource::from_raw_borrowed(&raw)?.GetDesc() };
        drop(shadow);
        let width = u32::try_from(desc.Width.checked_shr(mip)?.max(1)).ok()?;
        let height = desc.Height.checked_shr(mip)?.max(1);
        let (gate, gpu) = gate_probe::Lease::new(p.generation, p.index, width, height, describe)?;
        p.gate = Some(gate);
        Some(gpu)
    })
}

pub(super) fn dispatched(list: usize, groups: [u32; 3]) {
    if shader().is_none() { return }
    let Some(mut p) = BIND.replace(None) else { return };
    if p.list != list || p.generation != depth_lineage_queue::recording(list) || groups.contains(&0) || !super::list_x::depth_copy_unpredicated(list) { return }
    p.groups = groups;
    if let Some(t) = TARGET.lock().unwrap().as_mut().filter(|t| t.matches(&p)) {
        t.producers.retain(|x| x.list != list);
        if t.producers.len() < 32 { t.producers.push(p); }
    }
}

pub(super) fn take(list: usize) -> Option<(usize, Arc<Shadow>, u32, u32, u32, u64, u32, [u32; 3], Option<gate_probe::Lease>)> {
    shader()?;
    if !super::depth_capture::twin_due() { return None }
    let generation = depth_lineage_queue::recording(list);
    let mut g = TARGET.lock().unwrap();
    let t = g.as_mut()?;
    let at = t.producers.iter().position(|p| p.list == list)?;
    let p = t.producers.swap_remove(at);
    if p.generation != generation || !t.matches(&p) { return None }
    Some((t.owner, t.shadow.upgrade()?, t.mip, t.slice, t.sub, generation, p.index, p.groups, p.gate))
}
