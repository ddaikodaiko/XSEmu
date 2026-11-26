//! Predicate sources and query destinations in title memory that the GPU page directory maps:
//! read or written on the GPU through the recording's locator, so they reach the same owner the
//! translated shaders do. Ranges the directory has no entry for, and every refusal (counted),
//! keep the native path.

use super::{directory_touched, helper_context, live, shared_pipeline, title_range_touched, Copy, Operand};
use crate::hle::d3d12x::physical_directory::recordings;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

fn refused(what: &str, counter: &AtomicU64, why: String) {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: {what} of a directory-mapped title range kept on the native path: {why}; {n} so far");
    }
}

/// The 8 bytes of a predicate at title address `va`, gathered on `list` through its locator
/// after the list's earlier writers, in scratch the recording keeps (UNORDERED_ACCESS). None: the
/// directory has no entry there, or the gather is refused; the caller reads unified memory.
pub(crate) unsafe fn predicate(list: *mut c_void, va: u64) -> Option<ID3D12Resource> {
    static REFUSED: AtomicU64 = AtomicU64::new(0);
    if !directory_touched(va, 8) {
        return None;
    }
    let gathered = || -> Result<ID3D12Resource, String> {
        let ctx = helper_context(list)?;
        let read = Copy::read(va, 8, ctx.locator, live::uma_page).ok_or("a predicate page has no unified-memory view")?;
        let pipeline = shared_pipeline(&ctx.device)?;
        let (plan, data) = read.storage_bytes();
        let storage = Arc::new((live::scratch(&ctx.device, plan)?, live::scratch(&ctx.device, data)?));
        recordings::retain_native(list as usize, Arc::new((storage.clone(), pipeline.clone(), ctx.views.clone())))?;
        live::with_helper(&ctx.list, &ctx.scope, || pipeline.record_read(&ctx.list, &ctx.views, &read, &storage.0, &storage.1).map_err(String::from))?;
        super::super::transition(list, storage.1.as_raw() as usize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        Ok(ID3D12Resource::clone(&storage.1))
    };
    gathered().map_err(|why| refused("a predicate", &REFUSED, why)).ok()
}

/// ResolveQueryData of `count` queries (`bytes` of results, `visible` the occlusion results XCL
/// writes over them) into title buffer `dest` at `offset`: resolved into scratch, then written
/// through `list`'s locator to each destination page's one owner. false: the caller resolves
/// natively (the directory has no entry there, or a refusal).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn resolve(list: *mut c_void, heap: usize, ty: D3D12_QUERY_TYPE, first: u32, count: u32, dest: usize, offset: u64, bytes: u64, visible: &[(u32, u64)]) -> bool {
    static REFUSED: AtomicU64 = AtomicU64::new(0);
    if !title_range_touched(dest, offset, bytes) {
        return false;
    }
    let Some((op, va)) = Operand::retain(dest, offset).and_then(|op| op.guest_address.map(|va| (op, va))) else { return false };
    let recorded = || -> Result<(), String> {
        let ctx = helper_context(list)?;
        if ctx.scope.predicated() {
            return Err("the resolve is predicated".into());
        }
        let mut leases = Vec::new();
        let copy = Copy::write(va, bytes, ctx.locator, |page| {
            let (meta, lease) = live::fallback(&op.resource, va, op.native_offset, page)?;
            leases.push(lease);
            Some(meta)
        })
        .ok_or("a destination page has no proven owner")?;
        let pipeline = shared_pipeline(&ctx.device)?;
        let (plan, data) = copy.storage_bytes();
        let storage = Arc::new((live::scratch(&ctx.device, plan)?, live::scratch(&ctx.device, data)?));
        recordings::retain_native(list as usize, Arc::new((leases, storage.clone(), pipeline.clone(), ctx.views.clone(), op.resource.clone())))?;
        let results = storage.1.as_raw();
        type R = unsafe extern "system" fn(*mut c_void, usize, D3D12_QUERY_TYPE, u32, u32, *mut c_void, u64);
        super::super::pc::<R>(list, super::super::PC_RESOLVE_QUERY_DATA)(list, heap, ty, first, count, results, 0);
        if !visible.is_empty() {
            let at = storage.1.GetGPUVirtualAddress();
            let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = visible
                .iter()
                .flat_map(|&(i, v)| {
                    let a = at + 8 * u64::from(i - first);
                    [D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a, Value: v as u32 }, D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: a + 4, Value: (v >> 32) as u32 }]
                })
                .collect();
            let modes = vec![D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT; params.len()];
            type W = unsafe extern "system" fn(*mut c_void, u32, *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, *const D3D12_WRITEBUFFERIMMEDIATE_MODE);
            super::super::pc::<W>(list, super::super::PC_WRITE_BUFFER_IMMEDIATE)(list, params.len() as u32, params.as_ptr(), modes.as_ptr());
        }
        super::super::transition(list, results as usize, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        live::with_helper(&ctx.list, &ctx.scope, || pipeline.record_write(&ctx.list, &ctx.views, &copy, &storage.0, &storage.1).map_err(String::from))
    };
    recorded().map_err(|why| refused("ResolveQueryData", &REFUSED, why)).is_ok()
}
