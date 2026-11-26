use std::ffi::c_void;
use std::io::{BufWriter, Write};
use std::mem::{size_of, transmute_copy, ManuallyDrop};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::material_gate_capture as gate_probe;

static CLAIMED: AtomicBool = AtomicBool::new(false);
static TWIN_CLAIMED: AtomicBool = AtomicBool::new(false);
const MAX_BYTES: u64 = 32 << 20;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_DEPTH_CAPTURE").is_ok_and(|s| s.trim() == "1")) && super::gpu_timing::enabled()
}

pub(super) fn due() -> bool {
    enabled() && !CLAIMED.load(Ordering::Acquire)
}

pub(super) fn twin_due() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    enabled()
        && *ON.get_or_init(|| std::env::var("XCL_DEPTH_TWIN_EGRESS").is_ok_and(|s| s.trim() == "1"))
        && !TWIN_CLAIMED.load(Ordering::Acquire)
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Before {
    Legacy(u32),
    Enhanced { layout: i32, access: u32, sync: u32 },
}

impl From<D3D12_RESOURCE_STATES> for Before {
    fn from(s: D3D12_RESOURCE_STATES) -> Self {
        Self::Legacy(s.0 as u32)
    }
}

#[derive(Clone, Copy, Debug)]
enum Codes {
    F32,
    D16,
    D24,
}

impl Codes {
    fn bytes(self) -> usize {
        if matches!(self, Self::D16) {
            2
        } else {
            4
        }
    }

    fn code(self, p: &[u8]) -> u32 {
        if matches!(self, Self::D16) {
            return u16::from_le_bytes([p[0], p[1]]) as u32;
        }
        let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        if matches!(self, Self::D24) {
            v & 0xff_ffff
        } else {
            v
        }
    }

    fn value(self, code: u32) -> f64 {
        if matches!(self, Self::F32) {
            f32::from_bits(code) as f64
        } else {
            code as f64
        }
    }

    fn clear_value(self, clear: f32) -> f64 {
        match self {
            Self::F32 => clear as f64,
            Self::D16 => (clear as f64 * 65535.).round_ties_even(),
            Self::D24 => (clear as f64 * 16777215.).round_ties_even(),
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Dsv,
    Twin,
}

pub(super) struct Capture {
    kind: Kind,
    buffer: ID3D12Resource,
    _source: ID3D12Resource,
    width: usize,
    height: usize,
    pitch: usize,
    size: usize,
    codes: Codes,
    clear: Option<f32>,
    context: String,
    gate: Option<gate_probe::Lease>,
    dir: PathBuf,
    flight: AtomicUsize,
    generation: AtomicUsize,
}

impl Drop for Capture {
    fn drop(&mut self) {
        let generation = self.generation.load(Ordering::Relaxed);
        if generation != 0 && self.first_flight().is_none() {
            crate::xlog_warn!("depth capture: recording {generation} reset/released without submission; rearmed");
            match self.kind {
                Kind::Dsv => &CLAIMED,
                Kind::Twin => &TWIN_CLAIMED,
            }
            .store(false, Ordering::Release);
        }
    }
}

impl Capture {
    pub(super) fn submitted(&self, flight: usize) {
        let _ = self.flight.compare_exchange(0, flight, Ordering::AcqRel, Ordering::Acquire);
    }

    pub(super) fn first_flight(&self) -> Option<usize> {
        let v = self.flight.load(Ordering::Acquire);
        (v != 0).then_some(v)
    }

    pub(super) unsafe fn retired(self: &Arc<Self>, depth_generation: usize, q: usize, seq: u64) {
        let mut ptr = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: self.size };
        if self.buffer.Map(0, Some(&range), Some(&mut ptr)).is_err() || ptr.is_null() {
            crate::xlog_warn!("depth capture: could not map retired copy, generation={depth_generation} q={q} seq={seq}");
            return;
        }
        let row = self.width * self.codes.bytes();
        let mut pixels = Vec::with_capacity(row * self.height);
        for y in 0..self.height {
            pixels.extend_from_slice(std::slice::from_raw_parts(ptr.cast::<u8>().add(y * self.pitch), row));
        }
        self.buffer.Unmap(0, Some(&D3D12_RANGE::default()));
        if let Some(gate) = &self.gate {
            gate.report_retired(depth_generation, q, seq, [self.width, self.height], &self.context);
        }
        let capture = self.clone();
        let result = std::thread::Builder::new().name("xcl-depth-report".into()).spawn(move || {
            if let Err(e) = capture.report(depth_generation, q, seq, &pixels) {
                crate::xlog_warn!("depth capture: report failed for generation={depth_generation} q={q} seq={seq}: {e}");
            }
        });
        if let Err(e) = result {
            crate::xlog_warn!("depth capture: cannot start report writer: {e}");
        }
    }

    fn report(&self, depth_generation: usize, q: usize, seq: u64, pixels: &[u8]) -> std::io::Result<()> {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
        let prefix = match self.kind {
            Kind::Dsv => "depth-capture",
            Kind::Twin => "depth-twin-capture",
        };
        let file = self.dir.join(format!("{prefix}-{stamp}-{}-{depth_generation}", std::process::id()));
        std::fs::write(file.with_extension("raw"), pixels)?;
        let mut rows = BufWriter::new(std::fs::File::create(file.with_extension("tsv"))?);
        let clear = self.clear.map(|v| self.codes.clear_value(v));
        writeln!(rows, "# generation={depth_generation} q={q} seq={seq} width={} height={} codes={:?} endianness=little pitch={} raw_row_bytes={} clear_observed={:?} clear_in_row_units={clear:?} {}",
            self.width, self.height, self.codes, self.pitch, self.width * self.codes.bytes(), self.clear, self.context)?;
        writeln!(rows, "y\tfinite\tfinite_min\tfinite_max\tnonclear_finite\tnonclear_min\tnonclear_max\tnonclear_x_first\tnonclear_x_last")?;
        let mut codes = Vec::with_capacity(self.width * self.height);
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        let mut changed = 0usize;
        for (y, row) in pixels.chunks_exact(self.width * self.codes.bytes()).enumerate() {
            let (mut n, mut nc) = (0usize, 0usize);
            let (mut lo, mut hi, mut clo, mut chi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            let (mut xmin, mut xmax) = (usize::MAX, 0);
            for (x, p) in row.chunks_exact(self.codes.bytes()).enumerate() {
                let code = self.codes.code(p);
                codes.push(code);
                let v = self.codes.value(code);
                if !v.is_finite() {
                    continue;
                }
                n += 1;
                lo = lo.min(v);
                hi = hi.max(v);
                if clear.is_some_and(|c| c != v) {
                    nc += 1;
                    clo = clo.min(v);
                    chi = chi.max(v);
                    xmin = xmin.min(x);
                    xmax = xmax.max(x);
                }
            }
            let extrema = |n, lo, hi| if n == 0 { "NA\tNA".into() } else { format!("{lo}\t{hi}") };
            let comparison = if clear.is_none() {
                "NA\tNA\tNA\tNA\tNA".into()
            } else if nc == 0 {
                "0\tNA\tNA\tNA\tNA".into()
            } else {
                let (x0, y0, x1, _) = bounds.unwrap_or((xmin, y, xmax, y));
                bounds = Some((x0.min(xmin), y0, x1.max(xmax), y));
                changed += nc;
                format!("{nc}\t{clo}\t{chi}\t{xmin}\t{xmax}")
            };
            writeln!(rows, "{y}\t{n}\t{}\t{comparison}", extrema(n, lo, hi))?;
        }
        rows.flush()?;
        codes.sort_unstable();
        let mut counts = BufWriter::new(std::fs::File::create(file.with_extension("counts"))?);
        let mut unique = 0;
        let mut most = (0, 0);
        for run in codes.chunk_by(|a, b| a == b) {
            unique += 1;
            if run.len() > most.1 {
                most = (run[0], run.len());
            }
            counts.write_all(&run[0].to_le_bytes())?;
            counts.write_all(&(run.len() as u32).to_le_bytes())?;
        }
        counts.flush()?;
        crate::xlog_info!("depth capture: retired generation={depth_generation} q={q} seq={seq}; {}x{} {:?}, observed clear {:?}, finite nonclear count {:?} inclusive xy bounds {:?}, unique codes {unique} dominant 0x{:08x} ({}) grid={} rows=.tsv histogram=.counts (LE code:u32,count:u32); {}",
            self.width,self.height,self.codes,self.clear,clear.map(|_|changed),bounds,most.0,most.1,file.with_extension("raw").display(),self.context);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn record(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    dsv_format: DXGI_FORMAT,
    mip: u32,
    slice: u32,
    plane: u32,
    before: impl Into<Before>,
    clear: Option<f32>,
    context: String,
) -> Result<(), String> {
    if !enabled() || CLAIMED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("depth capture disabled or already claimed".into());
    }
    let result = (|| {
        let raw = list.as_raw();
        let before = before.into();
        let desc = resource.GetDesc();
        if list.GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT || !super::list_x::depth_copy_unpredicated(raw as usize) {
            return Err("depth copy needs a tracked unpredicated direct list".into());
        }
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.SampleDesc.Count != 1
            || plane != 0
            || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            || mip >= u32::from(desc.MipLevels)
            || mip >= 32
            || slice >= u32::from(desc.DepthOrArraySize)
            || clear.is_some_and(|v| !(0.0..=1.0).contains(&v))
        {
            return Err("not a known single-sample 2D depth plane, valid slice/mip or clear".into());
        }
        let codes = match (dsv_format, desc.Format) {
            (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_R16_TYPELESS) => Codes::D16,
            (DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D24_UNORM_S8_UINT | DXGI_FORMAT_R24G8_TYPELESS) => Codes::D24,
            (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_TYPELESS)
            | (DXGI_FORMAT_D32_FLOAT_S8X24_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT | DXGI_FORMAT_R32G8X24_TYPELESS) => Codes::F32,
            _ => return Err(format!("incompatible DSV/resource formats {}/{}", dsv_format.0, desc.Format.0)),
        };
        let list7 = match before {
            Before::Legacy(bits) => {
                let state = D3D12_RESOURCE_STATES(bits as i32);
                let reads = D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                if state != D3D12_RESOURCE_STATE_DEPTH_WRITE && (!state.contains(D3D12_RESOURCE_STATE_DEPTH_READ) || bits & !(reads.0 as u32) != 0) {
                    return Err(format!("legacy state {bits:#x} is not a known DSV state"));
                }
                None
            }
            Before::Enhanced { layout, access, sync } => {
                let a = D3D12_BARRIER_ACCESS(access as i32);
                let readable = D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ | D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
                let depth_sync = D3D12_BARRIER_SYNC_DEPTH_STENCIL | D3D12_BARRIER_SYNC_DRAW | D3D12_BARRIER_SYNC_ALL;
                let ok = match D3D12_BARRIER_LAYOUT(layout) {
                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ => a.contains(D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ) && a.0 & !readable.0 == 0,
                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE => a == D3D12_BARRIER_ACCESS_DEPTH_STENCIL_WRITE,
                    _ => false,
                };
                if !ok || sync & depth_sync.0 as u32 == 0 {
                    return Err(format!("unsupported enhanced depth layout={layout} access={access:#x} sync={sync:#x}"));
                }
                Some(list.cast::<ID3D12GraphicsCommandList7>().map_err(|e| e.to_string())?)
            }
        };
        record_copy(raw, resource, desc, mip, slice, plane, dsv_format, before, list7.as_ref(), codes, clear, Kind::Dsv, context, None)
    })();
    if result.is_err() {
        CLAIMED.store(false, Ordering::Release);
    }
    result
}

/// Records only the requested subresource of the private twin.
///
/// # Safety
///
/// The caller must identify the selected owner's twin and guarantee that this
/// subresource is currently in the legacy UNORDERED_ACCESS state on this list.
pub(super) unsafe fn record_twin(
    list: &ID3D12GraphicsCommandList,
    twin: &ID3D12Resource,
    mip: u32,
    slice: u32,
    context: String,
    gate: Option<gate_probe::Lease>,
) -> Result<(), String> {
    if gate_probe::enabled() && gate.is_none() {
        return Err("waiting for a dispatch carrying the isolated material-gate counters".into());
    }
    if !twin_due() || TWIN_CLAIMED.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Err("depth twin capture disabled or already claimed".into());
    }
    let result = (|| {
        let raw = list.as_raw();
        if list.GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT || !super::list_x::depth_copy_unpredicated(raw as usize) {
            return Err("depth twin copy needs a tracked unpredicated direct list".into());
        }
        let desc = twin.GetDesc();
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.SampleDesc.Count != 1
            || desc.Format != DXGI_FORMAT_R32_TYPELESS
            || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            || mip >= u32::from(desc.MipLevels)
            || mip >= 32
            || slice >= u32::from(desc.DepthOrArraySize)
        {
            return Err("not a single-sample 2D R32_TYPELESS UAV twin or valid slice/mip".into());
        }
        record_copy(
            raw, twin, desc, mip, slice, 0, DXGI_FORMAT_R32_FLOAT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS.into(), None, Codes::F32, None,
            Kind::Twin, format!("source=depth-uav-twin {context}"), gate,
        )
    })();
    if result.is_err() {
        TWIN_CLAIMED.store(false, Ordering::Release);
    }
    result
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_copy(
    raw: *mut c_void,
    resource: &ID3D12Resource,
    desc: D3D12_RESOURCE_DESC,
    mip: u32,
    slice: u32,
    plane: u32,
    view_format: DXGI_FORMAT,
    before: Before,
    list7: Option<&ID3D12GraphicsCommandList7>,
    codes: Codes,
    clear: Option<f32>,
    kind: Kind,
    context: String,
    gate: Option<gate_probe::Lease>,
) -> Result<(), String> {
    let width = (desc.Width >> mip).max(1);
    let height = (desc.Height >> mip).max(1);
    if width > 8192 || height > 8192 {
        return Err("depth subresource exceeds dimension cap".into());
    }
    let sub = mip + slice * u32::from(desc.MipLevels);
    let mut dev: Option<ID3D12Device> = None;
    resource.GetDevice(&mut dev).map_err(|e| e.to_string())?;
    let dev = dev.ok_or("no resource device")?;
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0, 0, 0);
    dev.GetCopyableFootprints(&desc, sub, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    let pitch = fp.Footprint.RowPitch as u64;
    if fp.Offset != 0
        || fp.Footprint.Width as u64 != width
        || fp.Footprint.Height != height
        || fp.Footprint.Depth != 1
        || rows != height
        || row_bytes != width * codes.bytes() as u64
        || pitch < row_bytes
        || total > MAX_BYTES
        || total < (height as u64 - 1) * pitch + row_bytes
    {
        return Err(format!(
            "unsupported depth footprint fmt={} dimensions={}x{} rows={rows} pitch={pitch} row_bytes={row_bytes} total={total}",
            fp.Footprint.Format.0, fp.Footprint.Width, fp.Footprint.Height
        ));
    }
    let dir = crate::state::boot().ok_or("no title logs directory")?.logs_dir();
    let mut buffer: Option<ID3D12Resource> = None;
    super::title_textures::own_work(|| {
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut buffer,
        )
    })
    .map_err(|e| e.to_string())?;
    let cap = Arc::new(Capture {
        kind,
        buffer: buffer.ok_or("no readback resource")?,
        _source: resource.clone(),
        width: width as usize,
        height: height as usize,
        pitch: pitch as usize,
        size: total as usize,
        codes,
        clear,
        dir,
        flight: AtomicUsize::new(0),
        generation: AtomicUsize::new(0),
        gate,
        context: format!(
            "resource={:p} rfmt={} vfmt={} fpfmt={} mip={mip} slice={slice} plane={plane} sub={sub} before={before:?} {context}",
            resource.as_raw(),
            desc.Format.0,
            view_format.0,
            fp.Footprint.Format.0
        ),
    });
    let depth_generation = super::gpu_timing::attach_depth(raw, cap.clone())?;
    cap.generation.store(depth_generation, Ordering::Relaxed);
    let barrier = |restore: bool| match before {
        Before::Legacy(bits) => {
            let (mut src, mut dst) = (D3D12_RESOURCE_STATES(bits as i32), D3D12_RESOURCE_STATE_COPY_SOURCE);
            if restore {
                std::mem::swap(&mut src, &mut dst);
            }
            let b = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: transmute_copy(resource),
                        Subresource: sub,
                        StateBefore: src,
                        StateAfter: dst,
                    }),
                },
                ..Default::default()
            };
            super::gpu_timing::native::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_RESOURCE_BARRIER)>(raw, 26)(raw, 1, &b);
        }
        Before::Enhanced { layout, access, sync } => {
            let list7 = list7.unwrap().as_raw();
            let (mut a, mut b) = (
                (D3D12_BARRIER_LAYOUT(layout), D3D12_BARRIER_ACCESS(access as i32), D3D12_BARRIER_SYNC(sync as i32)),
                (D3D12_BARRIER_LAYOUT_COPY_SOURCE, D3D12_BARRIER_ACCESS_COPY_SOURCE, D3D12_BARRIER_SYNC_COPY),
            );
            if restore {
                std::mem::swap(&mut a, &mut b);
            }
            let transition = D3D12_TEXTURE_BARRIER {
                SyncBefore: a.2,
                SyncAfter: b.2,
                AccessBefore: a.1,
                AccessAfter: b.1,
                LayoutBefore: a.0,
                LayoutAfter: b.0,
                pResource: transmute_copy(resource),
                Subresources: D3D12_BARRIER_SUBRESOURCE_RANGE {
                    IndexOrFirstMipLevel: mip,
                    NumMipLevels: 1,
                    FirstArraySlice: slice,
                    NumArraySlices: 1,
                    FirstPlane: plane,
                    NumPlanes: 1,
                },
                Flags: D3D12_TEXTURE_BARRIER_FLAG_NONE,
            };
            let group = D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_TEXTURE,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 { pTextureBarriers: &transition },
            };
            let slot = std::mem::offset_of!(ID3D12GraphicsCommandList7_Vtbl, Barrier) / size_of::<usize>();
            super::gpu_timing::native::<unsafe extern "system" fn(*mut c_void, u32, *const D3D12_BARRIER_GROUP)>(list7, slot)(list7, 1, &group);
        }
    };
    let dst = D3D12_TEXTURE_COPY_LOCATION {
        pResource: transmute_copy(&cap.buffer),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
    };
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: transmute_copy(resource),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
    };
    barrier(false);
    super::gpu_timing::native::<
        unsafe extern "system" fn(
            *mut c_void,
            *const D3D12_TEXTURE_COPY_LOCATION,
            u32,
            u32,
            u32,
            *const D3D12_TEXTURE_COPY_LOCATION,
            *const D3D12_BOX,
        ),
    >(raw, 16)(raw, &dst, 0, 0, 0, &src, std::ptr::null());
    barrier(true);
    crate::xlog_info!("depth capture: recorded generation={depth_generation} list={raw:p} {}; waiting for queue retirement", cap.context);
    Ok(())
}
