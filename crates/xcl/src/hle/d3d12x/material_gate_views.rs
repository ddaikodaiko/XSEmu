use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

use super::console_heap::{self, SLOT, Slot};
pub(super) mod writer;
pub(super) mod prepass;

const MAX_VIEWS: usize = 1 << 18;
const MAX_INPUTS: usize = 8;

#[derive(Clone, Copy, Debug)]
struct Subresources {
    mip: u32,
    mips: u32,
    first_slice: u32,
    slices: u32,
    plane: u32,
}

#[derive(Clone, Copy)]
pub(super) enum GuestView {
    Srv(i32),
    Uav(i32),
}

#[derive(Clone, Copy)]
pub(super) struct NativeView {
    owner: usize,
    resource: usize,
    requested: Option<GuestView>,
    view_format: i32,
    resource_format: i32,
    dimension: i32,
    flags: i32,
    mapping: u32,
    min_lod: f32,
    subresources: Subresources,
}

impl NativeView {
    pub(super) unsafe fn new(
        owner: usize, resource: usize, rd: &D3D12_RESOURCE_DESC, view: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
        requested: Option<GuestView>,
    ) -> Option<Self> {
        let mut r = Subresources { mip: 0, mips: rd.MipLevels.into(), first_slice: 0, slices: 1, plane: 0 };
        let (view_format, dimension, mapping, min_lod) = if let Some(v) = view {
            let a = &v.Anonymous;
            let min_lod = match v.ViewDimension {
                D3D12_SRV_DIMENSION_TEXTURE1D => {
                    let d = a.Texture1D;
                    (r.mip, r.mips) = (d.MostDetailedMip, d.MipLevels);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURE1DARRAY => {
                    let d = a.Texture1DArray;
                    (r.mip, r.mips, r.first_slice, r.slices) = (d.MostDetailedMip, d.MipLevels, d.FirstArraySlice, d.ArraySize);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURE2D => {
                    let d = a.Texture2D;
                    (r.mip, r.mips, r.plane) = (d.MostDetailedMip, d.MipLevels, d.PlaneSlice);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                    let d = a.Texture2DArray;
                    (r.mip, r.mips, r.first_slice, r.slices, r.plane) = (d.MostDetailedMip, d.MipLevels, d.FirstArraySlice, d.ArraySize, d.PlaneSlice);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURE2DMS => {
                    r.mips = 1;
                    0.0
                }
                D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY => {
                    let d = a.Texture2DMSArray;
                    (r.mips, r.first_slice, r.slices) = (1, d.FirstArraySlice, d.ArraySize);
                    0.0
                }
                D3D12_SRV_DIMENSION_TEXTURE3D => {
                    let d = a.Texture3D;
                    (r.mip, r.mips) = (d.MostDetailedMip, d.MipLevels);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURECUBE => {
                    let d = a.TextureCube;
                    (r.mip, r.mips, r.slices) = (d.MostDetailedMip, d.MipLevels, 6);
                    d.ResourceMinLODClamp
                }
                D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                    let d = a.TextureCubeArray;
                    (r.mip, r.mips, r.first_slice, r.slices) = (d.MostDetailedMip, d.MipLevels, d.First2DArrayFace, d.NumCubes.checked_mul(6)?);
                    d.ResourceMinLODClamp
                }
                _ => return None,
            };
            (v.Format.0, v.ViewDimension, v.Shader4ComponentMapping, min_lod)
        } else {
            let dimension = match (rd.Dimension, rd.DepthOrArraySize > 1, rd.SampleDesc.Count > 1) {
                (D3D12_RESOURCE_DIMENSION_TEXTURE1D, false, _) => D3D12_SRV_DIMENSION_TEXTURE1D,
                (D3D12_RESOURCE_DIMENSION_TEXTURE1D, true, _) => D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, false, false) => D3D12_SRV_DIMENSION_TEXTURE2D,
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, true, false) => D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, false, true) => D3D12_SRV_DIMENSION_TEXTURE2DMS,
                (D3D12_RESOURCE_DIMENSION_TEXTURE2D, true, true) => D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
                (D3D12_RESOURCE_DIMENSION_TEXTURE3D, _, _) => D3D12_SRV_DIMENSION_TEXTURE3D,
                _ => return None,
            };
            if dimension != D3D12_SRV_DIMENSION_TEXTURE3D {
                r.slices = rd.DepthOrArraySize.into();
            }
            (rd.Format.0, dimension, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 0.0)
        };
        Some(Self { owner, resource, requested, view_format, resource_format: rd.Format.0, dimension: dimension.0,
            flags: rd.Flags.0, mapping, min_lod, subresources: r })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Key {
    Shared(u32),
    Local(usize),
}

impl Key {
    fn of(s: &Slot) -> Self {
        if s.shared_srv != 0 { Self::Shared(s.index) } else { Self::Local(s.at) }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    at: usize,
    words: [u32; 8],
    view: NativeView,
}

#[derive(Default)]
struct Views {
    observed: HashMap<usize, u64>,
    next: u64,
    by_key: BTreeMap<Key, Entry>,
    by_owner: HashMap<usize, HashSet<Key>>,
}

impl Views {
    fn remove(&mut self, key: Key) {
        if let Some(entry) = self.by_key.remove(&key) {
            if let Some(keys) = self.by_owner.get_mut(&entry.view.owner) {
                keys.remove(&key);
                if keys.is_empty() {
                    self.by_owner.remove(&entry.view.owner);
                }
            }
        }
    }

    fn put(&mut self, s: &Slot, words: &[u32], view: Option<NativeView>) {
        let key = Key::of(s);
        self.remove(key);
        if let Some((view, words)) = view.zip(<[u32; 8]>::try_from(words).ok())
            .filter(|(_, w)| w[0] == s.index && w[3] >> 28 >= 8 && self.by_key.len() < MAX_VIEWS)
        {
            self.by_key.insert(key, Entry { at: s.at, words, view });
            self.by_owner.entry(view.owner).or_default().insert(key);
        }
    }

    fn forget_range(&mut self, range: std::ops::Range<Key>) {
        let keys = self.by_key.range(range).map(|(&key, _)| key).collect::<Vec<_>>();
        for key in keys {
            self.remove(key);
        }
    }
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);

fn inputs() -> &'static [(usize, u32)] {
    static INPUTS: OnceLock<Vec<(usize, u32)>> = OnceLock::new();
    INPUTS.get_or_init(|| {
        std::env::var("XCL_MATERIAL_GATE_INPUTS").unwrap_or_default().split(',').filter_map(|value| {
            let (reg, offset) = value.trim().strip_prefix('s')?.split_once('+')?;
            let reg = reg.parse::<usize>().ok()?;
            let offset = match offset.strip_prefix("0x") {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => offset.parse::<u32>().ok()?,
            };
            (reg < 32 && offset % SLOT == 0).then_some((reg, offset))
        }).take(MAX_INPUTS).collect()
    })
}

pub(super) fn enabled() -> bool {
    !inputs().is_empty()
}

pub(super) fn note(s: &Slot, words: &[u32], view: Option<NativeView>) {
    let mut g = VIEWS.lock().unwrap();
    if let Some(views) = g.as_mut() {
        views.put(s, words, view);
    } else if view.is_some() {
        g.get_or_insert_with(Default::default).put(s, words, view);
    }
}

pub(super) fn copied(src: &Slot, words: &[u32; 8]) -> Option<NativeView> {
    let g = VIEWS.lock().unwrap();
    let entry = g.as_ref()?.by_key.get(&Key::of(src))?;
    (entry.at == src.at && entry.words == *words).then_some(entry.view)
}

pub(super) fn forget_heap(local: std::ops::Range<usize>, shared: impl Iterator<Item = (u32, u32)>) {
    if let Some(views) = VIEWS.lock().unwrap().as_mut() {
        views.forget_range(Key::Local(local.start)..Key::Local(local.end));
        for (start, count) in shared {
            views.forget_range(Key::Shared(start)..Key::Shared(start + count));
        }
    }
}

pub(super) fn forget_resource(owner: usize) {
    if let Some(views) = VIEWS.lock().unwrap().as_mut() {
        views.observed.remove(&owner);
        if let Some(keys) = views.by_owner.remove(&owner) {
            for key in keys {
                views.by_key.remove(&key);
            }
        }
    }
}

pub(super) fn describe(io: &[u32]) -> (Option<u32>, String) {
    let (index, mut description) = describe_slots(io, inputs(), false);
    if writer::enabled() {
        let (_, epoch) = super::present::counters();
        description.push_str(&format!("; consumer_bind_epoch={epoch}"));
    }
    (index, description)
}

fn describe_slots(io: &[u32], selectors: &[(usize, u32)], writer_slots: bool) -> (Option<u32>, String) {
    let mut first_depth_index = None;
    let description = selectors.iter().enumerate().map(|(i, &(reg, offset))| {
        let label = format!("s{reg}+{offset:#x}/slot{}", offset / SLOT);
        let Some(base) = io.get(reg).copied().filter(|&v| v != 0) else { return format!("{label}: unset"); };
        let Some((at, words, key)) = console_heap::material_input(base, offset) else {
            return format!("{label}: outside a live aligned view table");
        };
        let Some(image) = crate::gpu::desc::image(&words) else { return format!("{label}: {at:#x} nonimage"); };
        let guest = format!("{label}: table={base:#x} slot_va={at:#x} idx={} T#={words:08x?} {:?} {}x{} gfx10={} mip={} array={} tile={}",
            words[0], image.ty, image.width, image.height, image.format, image.base_level, image.base_array, image.swizzle_mode);
        let entry = key.and_then(|key| {
            let mut g = VIEWS.lock().unwrap();
            let views = g.as_mut()?;
            let entry = *views.by_key.get(&key)?;
            if entry.words != words || super::real::resource_release_pending(entry.view.owner) { return None; }
            let stamp = if writer::enabled() && (writer_slots || i == 0) {
                if !views.observed.contains_key(&entry.view.owner) {
                    views.next += 1;
                    views.observed.insert(entry.view.owner, views.next);
                }
                views.observed.get(&entry.view.owner).copied()
            } else { None };
            Some((entry, stamp))
        });
        let Some((Entry { at: source, view, .. }, stamp)) = entry else {
            return format!("{guest} requested_dxgi=unknown native_srv=unresolved");
        };
        let Subresources { mip, mips, first_slice, slices, plane } = view.subresources;
        if i == 0 && matches!(view.requested, Some(GuestView::Srv(f)) if f == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT.0)
            && view.view_format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT.0 && view.resource_format == view.view_format
            && view.dimension == D3D12_SRV_DIMENSION_TEXTURE2D.0 && (mip, mips, first_slice, slices, plane) == (0, 1, 0, 1, 0)
            && view.mapping == D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING && view.min_lod == 0.0
        {
            first_depth_index = Some(words[0]);
        }
        let requested = match view.requested {
            Some(GuestView::Srv(f)) => format!("requested_dxgi={f} requested_kind=srv"),
            Some(GuestView::Uav(f)) => format!("requested_dxgi={f} requested_kind=uav"),
            None => "requested_dxgi=unknown".into(),
        };
        let identity = stamp.map_or_else(String::new, |stamp| writer::identity(at, words[0], view.owner, view.resource, stamp));
        format!("{guest} {requested} native_srv(resource={:#x} owner={:#x} source={source:#x} view_format={} resource_format={} dimension={} plane={plane} mip={mip} mips={mips} first_slice={first_slice} slices={slices} resource_flags={:#x} mapping={:#x} min_lod={}){identity}",
            view.resource, view.owner, view.view_format, view.resource_format, view.dimension, view.flags, view.mapping, view.min_lod)
    }).collect::<Vec<_>>().join("; ");
    (first_depth_index, description)
}
