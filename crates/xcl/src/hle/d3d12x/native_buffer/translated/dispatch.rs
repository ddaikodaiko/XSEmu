use super::{Generator, Rows, Span};
use crate::hle::d3d12x::indirect::{self, IndirectInput, IndirectReads};
use crate::hle::d3d12x::list_x::native_buffer::{fixed::Lease, live, Pipeline, Read};
use crate::hle::d3d12x::{console_heap, physical_directory::recordings, real, title_textures};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// most bytes of generated rows (per-record io and argument table) one indirect call may take
const ROWS_LIMIT: u64 = 4 << 20;

struct Shared {
    device: ID3D12Device,
    generator: Arc<Generator>,
    gather: Arc<Pipeline>,
    signatures: Vec<(ID3D12RootSignature, D3D12_INDIRECT_ARGUMENT_TYPE, u32, ID3D12CommandSignature)>,
}

static SHARED: Mutex<Option<Shared>> = Mutex::new(None);

struct Gathered {
    read: Read,
    plan: live::Pooled,
    data: live::Pooled,
}

unsafe fn with_shared<T>(dev: &ID3D12Device, f: impl FnOnce(&mut Shared) -> Result<T, String>) -> Result<T, String> {
    let mut g = SHARED.lock().unwrap();
    if g.as_ref().is_none_or(|s| &s.device != dev) {
        let (generator, gather) = title_textures::own_work(|| Ok::<_, String>((Generator::new(dev)?, Pipeline::new(dev)?)))?;
        *g = Some(Shared { device: dev.clone(), generator: Arc::new(generator), gather: Arc::new(gather), signatures: Vec::new() });
    }
    f(g.as_mut().unwrap())
}

/// XCL's command signature for `model` on `abi`: each record's row as root UAV 0, then its action.
unsafe fn signature(dev: &ID3D12Device, abi: &ID3D12RootSignature, model: &Rows) -> Result<ID3D12CommandSignature, String> {
    with_shared(dev, |s| {
        let stride = model.output_stride();
        if let Some(e) = s.signatures.iter().find(|e| &e.0 == abi && e.1 == model.action && e.2 == stride) {
            return Ok(e.3.clone());
        }
        let made = title_textures::own_work(|| model.signature(dev, abi)).map_err(|e| e.to_string())?;
        s.signatures.push((abi.clone(), model.action, stride, made.clone()));
        Ok(made)
    })
}

/// XCL's records for one call, made on the GPU and in the INDIRECT_ARGUMENT state: `commands`
/// (each row's GPU address, then the action), the gathered GPU `count`, and the first row.
pub(crate) struct Generated {
    pub(crate) commands: Arc<live::Pooled>,
    pub(crate) count: Option<ID3D12Resource>,
    pub(crate) first_row: u64,
}

impl Generated {
    unsafe fn barriers(&self, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> Vec<D3D12_RESOURCE_BARRIER> {
        let mut b = vec![indirect::transition(self.commands.as_raw() as usize, before, after)];
        b.extend(self.count.iter().map(|c| indirect::transition(c.as_raw() as usize, before, after)));
        b
    }
}

/// Records on `list`, unpredicated and with the helper's heaps (restored by `live::with_helper`
/// to `scope`), the helper work of one translated ExecuteIndirect of at most `max` records of
/// `model`: the arguments and count gathered through the recording's page directory (`reads`),
/// one row per record (the io words and argument table the record's root arguments make of the
/// base `model.template`, in unified memory from `take`), and XCL's records, the dispatches or
/// draws capped as the sanitiser caps them. `leases` and the helper's own objects are retained by
/// the recording.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_rows(
    list: &ID3D12GraphicsCommandList,
    model: &Rows,
    max: u32,
    reads: IndirectReads,
    mut leases: Vec<Lease>,
    mut take: impl FnMut(usize) -> Option<usize>,
    scope: &live::Scope,
    gfx: bool,
) -> Result<Generated, String> {
    let ty = list.GetType();
    if ty != D3D12_COMMAND_LIST_TYPE_DIRECT && (gfx || ty != D3D12_COMMAND_LIST_TYPE_COMPUTE) {
        return Err("translated indirect rows need a direct list (or a compute list for dispatches)".into());
    }
    if gfx == (model.action == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH) {
        return Err("the rows' action does not match the pipeline".into());
    }
    let rows_bytes = u64::from(max) * u64::from(model.row_bytes);
    if max == 0 || rows_bytes > ROWS_LIMIT {
        return Err(format!("{max} records take {rows_bytes} bytes of rows (at most {ROWS_LIMIT})"));
    }
    let mut device: Option<ID3D12Device> = None;
    list.GetDevice(&mut device).map_err(|e| e.to_string())?;
    let dev = device.ok_or("the command list has no device")?;
    let views = console_heap::global_heap().ok_or("no shared view heap for the gather")?;
    let (generator, gather) = with_shared(&dev, |s| Ok((s.generator.clone(), s.gather.clone())))?;
    let gathered = |input: IndirectInput| -> Result<Arc<Gathered>, String> {
        let IndirectInput::Gather(read) = input else {
            return Err("an indirect input outside title memory is not gathered".into());
        };
        let (p, b) = read.storage_bytes();
        let (plan, data) = (live::scratch(&dev, p)?, live::scratch(&dev, b)?);
        if read.0.bind_storage(&plan, &data).is_none() {
            return Err("gather storage does not fit".into());
        }
        Ok(Arc::new(Gathered { read, plan, data }))
    };
    let arguments = gathered(reads.arguments)?;
    let count = reads.count.map(gathered).transpose()?;
    let template: Vec<u8> = model.template.iter().flat_map(|w| w.to_le_bytes()).collect();
    let template_at = take(template.len()).ok_or("unified memory is full")?;
    std::ptr::copy_nonoverlapping(template.as_ptr(), template_at as *mut u8, template.len());
    let rows_at = take(rows_bytes as usize).ok_or("unified memory is full")?;
    let window = |p: usize, n: usize| -> Result<(ID3D12Resource, u64), String> {
        let (raw, offset) = real::uma_window_buffer(p, n).ok_or("not in one memory window")?;
        let r = ID3D12Resource::from_raw_borrowed(&(raw as *mut c_void)).cloned().ok_or("no memory window buffer")?;
        Ok((r, offset))
    };
    let (template_buffer, template_offset) = window(template_at, template.len())?;
    let (rows_buffer, rows_offset) = window(rows_at, rows_bytes as usize)?;
    let commands = Arc::new(live::scratch(&dev, u64::from(max) * u64::from(model.output_stride()))?);
    let bindings = Generator::bindings(
        model,
        max,
        rows_at as u64,
        Span { resource: &template_buffer, offset: template_offset },
        Span { resource: &arguments.data, offset: 0 },
        count.as_ref().map(|c| Span { resource: &c.data, offset: 0 }),
        Span { resource: &rows_buffer, offset: rows_offset },
        Span { resource: &commands, offset: 0 },
    )
    .ok_or("generated rows do not fit their buffers")?;
    let generated = Generated { first_row: bindings.buffers[2], count: count.as_ref().map(|c| ID3D12Resource::clone(&c.data)), commands };
    leases.push(Arc::new((generator.clone(), gather.clone(), views.clone())));
    leases.push(Arc::new((template_buffer, rows_buffer, generated.commands.clone())));
    leases.push(Arc::new((arguments.clone(), count.clone())));
    recordings::retain_native(list.as_raw() as usize, Arc::new(leases))?;
    live::with_helper(list, scope, || {
        for g in [Some(&arguments), count.as_ref()].into_iter().flatten() {
            gather.record_read(list, &views, &g.read, &g.plan, &g.data)?;
        }
        generator.record(list, bindings, gfx);
        list.ResourceBarrier(&generated.barriers(D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT));
        Ok(())
    })?;
    Ok(generated)
}

/// Records on `list` one translated ExecuteIndirect of at most `max` records of `model`: the
/// helper work of `record_rows`, then the call on XCL's ABI (`abi`, `pso`, each record's row as
/// root UAV 0, `res` as root UAV 1; the graphics roots for `gfx`, after `before_call`) under the
/// list's own predication and heaps (`scope`). Err: nothing of the title's ran; only helper work
/// before the call may have been recorded.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn execute(
    list: &ID3D12GraphicsCommandList,
    model: &Rows,
    max: u32,
    reads: IndirectReads,
    mut leases: Vec<Lease>,
    take: impl FnMut(usize) -> Option<usize>,
    abi: &ID3D12RootSignature,
    pso: &ID3D12PipelineState,
    res: u64,
    scope: &live::Scope,
    gfx: bool,
    before_call: impl FnOnce(),
) -> Result<(), String> {
    let mut device: Option<ID3D12Device> = None;
    list.GetDevice(&mut device).map_err(|e| e.to_string())?;
    let signature = signature(&device.ok_or("the command list has no device")?, abi, model)?;
    leases.push(Arc::new((signature.clone(), abi.clone(), pso.clone())));
    let generated = record_rows(list, model, max, reads, leases, take, scope, gfx)?;
    if gfx {
        list.SetGraphicsRootSignature(abi);
        list.SetGraphicsRootUnorderedAccessView(0, generated.first_row);
        list.SetGraphicsRootUnorderedAccessView(1, res);
    } else {
        list.SetComputeRootSignature(abi);
        list.SetComputeRootUnorderedAccessView(0, generated.first_row);
        list.SetComputeRootUnorderedAccessView(1, res);
    }
    live::set_pipeline(list, pso);
    before_call();
    let execute: unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, u64, *mut c_void, u64) =
        std::mem::transmute(real::original_or_native(list.as_raw(), 59).0);
    execute(list.as_raw(), signature.as_raw(), max, generated.commands.as_raw(), 0, generated.count.as_ref().map_or(std::ptr::null_mut(), |c| c.as_raw()), 0);
    list.ResourceBarrier(&generated.barriers(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
    Ok(())
}
