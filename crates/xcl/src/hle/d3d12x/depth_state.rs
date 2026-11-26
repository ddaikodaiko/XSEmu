use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum State {
    Legacy(D3D12_RESOURCE_STATES),
    Enhanced { layout: D3D12_BARRIER_LAYOUT, access: D3D12_BARRIER_ACCESS, sync: D3D12_BARRIER_SYNC },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Range {
    Unknown,
    All,
    One(u32),
    Rect { ranges: [(u32, u32); 3], mips: u32, slices: u32 },
}

impl Range {
    fn flat(sub: u32) -> Self {
        if sub == u32::MAX { Self::All } else { Self::One(sub) }
    }

    unsafe fn enhanced(resource: usize, r: D3D12_BARRIER_SUBRESOURCE_RANGE) -> Option<Self> {
        if r.NumMipLevels == 0 {
            return Some(Self::flat(r.IndexOrFirstMipLevel));
        }
        let d = ID3D12Resource::from_raw_borrowed(&(resource as *mut c_void))?.GetDesc();
        let (mips, slices) = (u32::from(d.MipLevels), u32::from(d.DepthOrArraySize));
        if !matches!(d.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE1D | D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            return None;
        }
        Self::texture(r, mips, slices, 2)
    }

    pub(super) fn texture(r: D3D12_BARRIER_SUBRESOURCE_RANGE, mips: u32, slices: u32, planes: u32) -> Option<Self> {
        if r.NumMipLevels == 0 {
            let sub = r.IndexOrFirstMipLevel;
            return (sub == u32::MAX || sub < mips.checked_mul(slices)?.checked_mul(planes)?).then(|| Self::flat(sub));
        }
        if r.IndexOrFirstMipLevel.checked_add(r.NumMipLevels)? > mips
            || r.NumArraySlices == 0 || r.FirstArraySlice.checked_add(r.NumArraySlices)? > slices
            || r.NumPlanes == 0 || r.FirstPlane.checked_add(r.NumPlanes)? > planes
        {
            return None;
        }
        Some(Self::Rect {
            ranges: [(r.IndexOrFirstMipLevel, r.NumMipLevels), (r.FirstArraySlice, r.NumArraySlices), (r.FirstPlane, r.NumPlanes)],
            mips,
            slices,
        })
    }

    pub(super) fn covers(self, sub: u32) -> bool {
        match self {
            Self::Unknown => false,
            Self::All => true,
            Self::One(n) => sub == n,
            Self::Rect { ranges, mips, slices } => {
                let coords = [sub % mips, sub / mips % slices, sub / mips / slices];
                ranges.into_iter().zip(coords).all(|((first, count), i)| i.wrapping_sub(first) < count)
            }
        }
    }
}

#[derive(Default)]
struct Recording {
    resources: HashMap<usize, Vec<(Range, Option<State>)>>,
    pass: Option<D3D12_RENDER_PASS_FLAGS>,
    suspended: bool,
}

impl Recording {
    fn note(&mut self, resource: usize, range: Range, state: Option<State>) {
        if resource == 0 {
            self.resources.clear();
            return;
        }
        if state.is_none() && range == Range::All {
            self.resources.remove(&resource);
            return;
        }
        if state.is_none() && !self.resources.contains_key(&resource) {
            return;
        }
        if self.resources.len() >= 256 && !self.resources.contains_key(&resource) { return }
        let notes = self.resources.entry(resource).or_default();
        if range == Range::All || notes.len() >= 32 {
            notes.clear();
        }
        notes.retain(|(r, _)| *r != range);
        notes.push((range, state));
    }

    fn legacy(&mut self, kind: u32, flags: u32, resource: usize, sub: u32, after: D3D12_RESOURCE_STATES) {
        match kind {
            0 => {
                let read = D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE;
                let depth = after == D3D12_RESOURCE_STATE_DEPTH_WRITE
                    || after.0 & D3D12_RESOURCE_STATE_DEPTH_READ.0 != 0 && after.0 & !read.0 == 0;
                self.note(resource, Range::flat(sub), (flags == 0 && depth).then_some(State::Legacy(after)));
            }
            2 => {}
            _ => self.resources.clear(),
        }
    }
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Recording>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static FULL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn recording(map: &mut HashMap<usize, Recording>, list: usize) -> Option<&mut Recording> {
    if map.len() >= 512 && !map.contains_key(&list) {
        FULL.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    Some(map.entry(list).or_default())
}

pub(super) fn enabled() -> bool {
    super::depth_capture_views::enabled() && !FULL.load(std::sync::atomic::Ordering::Relaxed)
}

pub(super) fn proven(list: usize, resource: usize, subresource: u32) -> Option<State> {
    if !enabled() || subresource == u32::MAX {
        return None;
    }
    let g = RECORDINGS.lock().unwrap();
    let r = g.get(&list)?;
    if r.pass.is_some() || r.suspended {
        return None;
    }
    r.resources.get(&resource)?.iter().rev().find(|(range, _)| range.covers(subresource))?.1
}

pub(super) fn reset(list: usize) {
    if enabled() {
        RECORDINGS.lock().unwrap().remove(&list);
    }
}

pub(super) fn forget(resource: usize) {
    if enabled() {
        for r in RECORDINGS.lock().unwrap().values_mut() {
            r.resources.remove(&resource);
        }
    }
}

pub(super) fn render_pass(list: usize, flags: Option<D3D12_RENDER_PASS_FLAGS>) {
    if !enabled() {
        return;
    }
    let mut g = RECORDINGS.lock().unwrap();
    let Some(r) = recording(&mut g, list) else { return };
    if flags.is_none() {
        r.suspended |= r.pass.is_some_and(|f| f.contains(D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS));
        r.resources.clear();
    }
    r.pass = flags;
}

pub(super) fn invalidate(list: usize) {
    if enabled() {
        if let Some(r) = RECORDINGS.lock().unwrap().get_mut(&list) {
            r.resources.clear();
        }
    }
}

pub(super) unsafe fn native_discard(list: usize, resource: usize, desc: *const D3D12_DISCARD_REGION) {
    use super::depth_lineage as lineage;
    if !lineage::twin_enabled() || !lineage::interested(resource) { return }
    let Some(d) = desc.as_ref() else {
        return lineage::note(resource, list, Range::All, "native-discard", || "native DiscardResource, all subresources".into());
    };
    let detail = || format!("native DiscardResource, subresource start={} count={} rects={} (rectangles may be partial)", d.FirstSubresource, d.NumSubresources, d.NumRects);
    if let Some(end) = d.FirstSubresource.checked_add(d.NumSubresources).filter(|_| d.NumSubresources <= 16) {
        for sub in d.FirstSubresource..end { lineage::note(resource, list, Range::One(sub), "native-discard", detail); }
    } else {
        lineage::note(resource, list, Range::Unknown, "native-discard", detail);
    }
}

pub(super) fn guest_legacy_raw(list: usize, b: &[u32; 8]) {
    use super::depth_lineage as lineage;
    if !lineage::twin_enabled() || b[1] > 3 { return }
    let resource = (u64::from(b[2]) | u64::from(b[3]) << 32) as usize;
    match b[0] {
        0 if lineage::interested(resource) => {
            let range = Range::flat(b[4]);
            let detail = || format!("guest before={:#x} after={:#x} flags={:#x}", b[5], b[6], b[1]);
            lineage::note(resource, list, range, "guest-legacy-named", detail);
            for (flag, kind) in [
                (D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, "guest-legacy-split-begin"),
                (D3D12_RESOURCE_BARRIER_FLAG_END_ONLY, "guest-legacy-split-end"),
            ] {
                if b[1] & flag.0 as u32 != 0 { lineage::note(resource, list, range, kind, detail); }
            }
        }
        1 => {
            let after = (u64::from(b[4]) | u64::from(b[5]) << 32) as usize;
            for target in [resource, after] {
                if lineage::interested(target) {
                    lineage::note(target, list, Range::All, "guest-alias-barrier", || {
                        format!("guest before={resource:#x} after={after:#x} flags={:#x}", b[1])
                    });
                }
            }
        }
        _ => {}
    }
}

fn note_legacy(list: usize, resource: usize, sub: u32, before: u32, after: u32, flags: u32) {
    use super::depth_lineage as lineage;
    if !lineage::interested(resource) { return }
    let range = Range::flat(sub);
    let detail = || format!("recorded before={before:#x} after={after:#x} flags={flags:#x}");
    lineage::note(resource, list, range, "native-legacy-transition", detail);
    if lineage::twin_enabled() {
        for (flag, kind) in [
            (D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, "native-legacy-split-begin"),
            (D3D12_RESOURCE_BARRIER_FLAG_END_ONLY, "native-legacy-split-end"),
        ] {
            if flags & flag.0 as u32 != 0 {
                lineage::note(resource, list, range, kind, detail);
            }
        }
    }
}

fn note_alias(list: usize, before: usize, after: usize) {
    use super::depth_lineage as lineage;
    let detail = || format!("recorded before={before:#x} after={after:#x}");
    lineage::note(before, list, Range::All, "native-alias-barrier", detail);
    if after != before {
        lineage::note(after, list, Range::All, "native-alias-barrier", detail);
    }
    if after != 0 && lineage::twin_enabled() {
        lineage::note(after, list, Range::All, "native-alias-activate", detail);
    }
}

pub(super) fn legacy_raw(list: usize, barriers: &[[u32; 8]]) {
    if !enabled() || barriers.is_empty() {
        return;
    }
    let mut g = RECORDINGS.lock().unwrap();
    let Some(r) = recording(&mut g, list) else { return };
    for b in barriers {
        let resource = (u64::from(b[2]) | u64::from(b[3]) << 32) as usize;
        if b[0] == 0 {
            note_legacy(list, resource, b[4], b[5], b[6], b[1]);
        } else if b[0] == 1 && super::depth_lineage::enabled() {
            let other = (u64::from(b[4]) | u64::from(b[5]) << 32) as usize;
            note_alias(list, resource, other);
        }
        r.legacy(b[0], b[1], resource, b[4], D3D12_RESOURCE_STATES(b[6] as i32));
    }
}

pub(super) unsafe fn legacy(list: usize, n: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
    if !enabled() || n == 0 {
        return;
    }
    if barriers.is_null() {
        invalidate(list);
        return;
    }
    let mut g = RECORDINGS.lock().unwrap();
    let Some(r) = recording(&mut g, list) else { return };
    for b in std::slice::from_raw_parts(barriers, n as usize) {
        match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => {
                let t = &b.Anonymous.Transition;
                let resource = t.pResource.as_ref().map_or(0, |r| r.as_raw() as usize);
                note_legacy(list, resource, t.Subresource, t.StateBefore.0 as u32, t.StateAfter.0 as u32, b.Flags.0 as u32);
                r.legacy(b.Type.0 as u32, b.Flags.0 as u32, resource, t.Subresource, t.StateAfter);
            }
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => {
                if super::depth_lineage::enabled() {
                    let a = &b.Anonymous.Aliasing;
                    let before = a.pResourceBefore.as_ref().map_or(0, |r| r.as_raw() as usize);
                    let after = a.pResourceAfter.as_ref().map_or(0, |r| r.as_raw() as usize);
                    note_alias(list, before, after);
                }
                r.resources.clear();
            }
            D3D12_RESOURCE_BARRIER_TYPE_UAV => {}
            _ => r.resources.clear(),
        }
    }
}

pub(super) unsafe fn enhanced(list: usize, n: u32, groups: *const D3D12_BARRIER_GROUP) {
    if !enabled() || n == 0 {
        return;
    }
    if groups.is_null() {
        invalidate(list);
        return;
    }
    let mut g = RECORDINGS.lock().unwrap();
    let Some(r) = recording(&mut g, list) else { return };
    for b in std::slice::from_raw_parts(groups, n as usize) {
        if b.NumBarriers == 0 {
            continue;
        }
        match b.Type {
            D3D12_BARRIER_TYPE_GLOBAL | D3D12_BARRIER_TYPE_BUFFER => {}
            D3D12_BARRIER_TYPE_TEXTURE if !b.Anonymous.pTextureBarriers.is_null() => {
                for t in std::slice::from_raw_parts(b.Anonymous.pTextureBarriers, b.NumBarriers as usize) {
                    let resource = t.pResource.as_ref().map_or(0, |r| r.as_raw() as usize);
                    let read = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ;
                    let write = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE;
                    let depth = match t.LayoutAfter {
                        D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ => t.AccessAfter == read,
                        D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE =>
                            t.AccessAfter.contains(write) && t.AccessAfter.0 & !(read | write).0 == 0,
                        _ => false,
                    };
                    let unsplit = (t.SyncBefore.0 | t.SyncAfter.0) & D3D12_BARRIER_SYNC_SPLIT.0 == 0;
                    let state = (depth && unsplit && t.Flags == D3D12_TEXTURE_BARRIER_FLAG_NONE
                        && t.LayoutBefore != D3D12_BARRIER_LAYOUT_UNDEFINED)
                        .then_some(State::Enhanced { layout: t.LayoutAfter, access: t.AccessAfter, sync: t.SyncAfter });
                    let interested = super::depth_lineage::interested(resource);
                    let track = state.is_some() || r.resources.contains_key(&resource) || resource == 0;
                    if !interested && !track { continue }
                    let range = Range::enhanced(resource, t.Subresources);
                    if interested && (range.is_some() || super::depth_lineage::twin_enabled()) {
                        let range = range.unwrap_or(Range::Unknown);
                        let detail = || format!("recorded sync={:#x}->{:#x} access={:#x}->{:#x} layout={}->{} flags={:#x}",
                            t.SyncBefore.0, t.SyncAfter.0, t.AccessBefore.0, t.AccessAfter.0, t.LayoutBefore.0, t.LayoutAfter.0, t.Flags.0);
                        super::depth_lineage::note(resource, list, range, "native-texture-transition", detail);
                        if super::depth_lineage::twin_enabled() {
                            for (matches, kind) in [
                                (t.SyncAfter.contains(D3D12_BARRIER_SYNC_SPLIT), "native-texture-split-begin"),
                                (t.SyncBefore.contains(D3D12_BARRIER_SYNC_SPLIT), "native-texture-split-end"),
                                (t.Flags.contains(D3D12_TEXTURE_BARRIER_FLAG_DISCARD), "native-texture-discard"),
                                ((unsplit || t.SyncBefore.contains(D3D12_BARRIER_SYNC_SPLIT))
                                    && t.LayoutBefore == D3D12_BARRIER_LAYOUT_UNDEFINED
                                    && t.LayoutAfter != D3D12_BARRIER_LAYOUT_UNDEFINED, "native-texture-activation"),
                            ] {
                                if matches {
                                    super::depth_lineage::note(resource, list, range, kind, detail);
                                }
                            }
                        }
                    }
                    if track {
                        match range {
                            Some(range) => r.note(resource, range, state),
                            None => r.note(resource, Range::All, None),
                        }
                    }
                }
            }
            _ => r.resources.clear(),
        }
    }
}
