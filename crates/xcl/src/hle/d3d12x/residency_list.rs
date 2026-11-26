#![allow(non_snake_case)]

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::real::original_or_native;
use super::residency::{note_list_resources, note_list_untracked};

pub(super) const OPAQUE_METHODS: &[&str] = &[
    "DrawInstanced", "DrawIndexedInstanced", "Dispatch", "OMSetRenderTargets", "ExecuteBundle",
    "ClearDepthStencilView", "ClearRenderTargetView", "ClearUnorderedAccessViewUint",
    "ClearUnorderedAccessViewFloat", "ExecuteIndirect", "AtomicCopyBufferUINT",
    "AtomicCopyBufferUINT64", "WriteBufferImmediate", "BeginRenderPass", "InitializeMetaCommand",
    "ExecuteMetaCommand", "BuildRaytracingAccelerationStructure",
    "EmitRaytracingAccelerationStructurePostbuildInfo", "CopyRaytracingAccelerationStructure",
    "DispatchRays", "DispatchMesh", "Barrier", "SetProgram", "DispatchGraph", "WaitBarrier",
    "SignalBarrier", "CopyBufferRegions", "CopyTextureRegions", "CopyResources", "CopyTilesAsync",
    "ResolveSubresourceRegionAsync", "ResolveQueryDataAsync", "ClearBoundDepthStencilView",
    "ClearBoundRenderTargetViews", "ClearTextureSubresources", "FillBuffers",
];

type Obj = *mut c_void;

macro_rules! direct_uses {
    ($($method:ident $vtbl:ident ($this:ident $(, $arg:ident: $ty:ty)*) $class:ident => $record:expr $(, $track:expr)? $(=> $writes:expr)?;)*) => {
        pub(super) fn overrides(v: &mut Vec<(&'static str, usize)>) {
            let timing = cfg!(test) || super::gpu_timing::enabled();
            let residency = super::residency::eviction_enabled();
            let state = super::depth_state::enabled();
            let writeback = super::image_writeback::on();
            v.extend([$((stringify!($method), $method as usize)),*].into_iter().filter(|&(name, _)| {
                timing || state && matches!(name, "Barrier" | "ResourceBarrier")
                    || residency && !matches!(name, "Barrier" | "WriteBufferImmediate" | "ClearUnorderedAccessViewUint" | "ClearUnorderedAccessViewFloat")
                    || writeback && matches!(name, "CopyTextureRegion" | "CopyResource" | "CopyTiles" | "ResolveSubresource"
                        | "ResolveSubresourceRegion" | "ClearUnorderedAccessViewUint" | "ClearUnorderedAccessViewFloat")
            }));
        }

        $(unsafe extern "system" fn $method($this: Obj $(, $arg: $ty)*) {
            let slot = std::mem::offset_of!($vtbl, $method) / size_of::<usize>();
            let (target, adopted) = original_or_native($this, slot);
            let f: unsafe extern "system" fn(Obj $(, $ty)*) = std::mem::transmute(target);
            if !adopted {
                return f($this $(, $arg)*);
            }
            if super::residency::eviction_enabled() {
                $record;
            }
            $($writes;)?
            direct_uses!(@emit $class, $this, || f($this $(, $arg)*));
            $($track;)?
        })*
    };
    (@emit Untimed, $list:ident, $call:expr) => { ($call)() };
    (@emit $class:ident, $list:ident, $call:expr) => {
        super::gpu_timing::class($list, super::gpu_timing::Kind::$class, $call)
    };
}

direct_uses! {
    CopyBufferRegion ID3D12GraphicsCommandList_Vtbl(list, dst: Obj, dst_off: u64, src: Obj, src_off: u64, bytes: u64) Transfer
        => note_list_resources(list as usize, [dst as usize, src as usize]);
    CopyTextureRegion ID3D12GraphicsCommandList_Vtbl(list, dst: *const D3D12_TEXTURE_COPY_LOCATION, x: u32, y: u32, z: u32,
                      src: *const D3D12_TEXTURE_COPY_LOCATION, box_: *const D3D12_BOX) Transfer
        => note_list_resources(list as usize, [dst, src].into_iter().filter_map(|p|
            p.as_ref()?.pResource.as_ref().map(|r| r.as_raw() as usize)))
        => super::image_writeback::note_copy_region(list, dst, x, y, z, src, box_);
    CopyResource ID3D12GraphicsCommandList_Vtbl(list, dst: Obj, src: Obj) Transfer
        => note_list_resources(list as usize, [dst as usize, src as usize])
        => super::image_writeback::note_write(list, dst, super::image_writeback::Covers::Whole);
    CopyTiles ID3D12GraphicsCommandList_Vtbl(list, tiled: Obj, coord: *const D3D12_TILED_RESOURCE_COORDINATE,
              region: *const D3D12_TILE_REGION_SIZE, buf: Obj, offset: u64, flags: D3D12_TILE_COPY_FLAGS) Transfer
        => note_list_resources(list as usize, [tiled as usize, buf as usize])
        => if flags.0 & D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE.0 != 0 {
            super::image_writeback::note_write(list, tiled, super::image_writeback::Covers::Part)
        };
    ResolveSubresource ID3D12GraphicsCommandList_Vtbl(list, dst: Obj, dst_sub: u32, src: Obj, src_sub: u32, fmt: i32) Transfer
        => note_list_resources(list as usize, [dst as usize, src as usize])
        => super::image_writeback::note_write(list, dst, super::image_writeback::Covers::Sub(dst_sub));
    ResourceBarrier ID3D12GraphicsCommandList_Vtbl(list, n: u32, barriers: *const D3D12_RESOURCE_BARRIER) Barrier
        => note_barriers(list as usize, n, barriers), super::depth_state::legacy(list as usize, n, barriers);
    DiscardResource ID3D12GraphicsCommandList_Vtbl(list, res: Obj, desc: *const D3D12_DISCARD_REGION) Barrier
        => note_list_resources(list as usize, [res as usize]), super::depth_state::native_discard(list as usize, res as usize, desc);
    ResolveQueryData ID3D12GraphicsCommandList_Vtbl(list, heap: Obj, ty: D3D12_QUERY_TYPE, first: u32, n: u32, dst: Obj, offset: u64) Transfer
        => note_list_resources(list as usize, [dst as usize]);
    SetPredication ID3D12GraphicsCommandList_Vtbl(list, buf: Obj, offset: u64, op: D3D12_PREDICATION_OP) Untimed
        => note_list_resources(list as usize, [buf as usize]);
    ResolveSubresourceRegion ID3D12GraphicsCommandList1_Vtbl(list, dst: Obj, dst_sub: u32, x: u32, y: u32, src: Obj, src_sub: u32,
                             rect: *const windows::Win32::Foundation::RECT, fmt: i32, mode: D3D12_RESOLVE_MODE) Transfer
        => note_list_resources(list as usize, [dst as usize, src as usize])
        => super::image_writeback::note_write(list, dst, super::image_writeback::Covers::Part);
    RSSetShadingRateImage ID3D12GraphicsCommandList5_Vtbl(list, img: Obj) Untimed => note_list_resources(list as usize, [img as usize]);
    Barrier ID3D12GraphicsCommandList7_Vtbl(list, n: u32, groups: *const D3D12_BARRIER_GROUP) Barrier
        => note_list_untracked(list as usize), super::depth_state::enhanced(list as usize, n, groups);
    WriteBufferImmediate ID3D12GraphicsCommandList2_Vtbl(list, n: u32, args: *const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER, modes: *const D3D12_WRITEBUFFERIMMEDIATE_MODE) Transfer
        => note_list_untracked(list as usize);
    ClearUnorderedAccessViewUint ID3D12GraphicsCommandList_Vtbl(list, gpu: D3D12_GPU_DESCRIPTOR_HANDLE, cpu: D3D12_CPU_DESCRIPTOR_HANDLE, res: Obj,
                                values: *const u32, n: u32, rect: *const windows::Win32::Foundation::RECT) Clear
        => note_list_resources(list as usize, [res as usize])
        => super::image_writeback::note_write(list, res, super::image_writeback::Covers::Part);
    ClearUnorderedAccessViewFloat ID3D12GraphicsCommandList_Vtbl(list, gpu: D3D12_GPU_DESCRIPTOR_HANDLE, cpu: D3D12_CPU_DESCRIPTOR_HANDLE, res: Obj,
                                 values: *const f32, n: u32, rect: *const windows::Win32::Foundation::RECT) Clear
        => note_list_resources(list as usize, [res as usize])
        => super::image_writeback::note_write(list, res, super::image_writeback::Covers::Part);
}

unsafe fn note_barriers(list: usize, n: u32, barriers: *const D3D12_RESOURCE_BARRIER) {
    if n == 0 || barriers.is_null() {
        return;
    }
    let mut opaque = false;
    let resources = std::slice::from_raw_parts(barriers, n as usize).iter().flat_map(|b| {
        let raw = |r: Option<&ID3D12Resource>| r.map_or(0, |r| r.as_raw() as usize);
        match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => [raw(b.Anonymous.Transition.pResource.as_ref()), 0],
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => {
                let a = &b.Anonymous.Aliasing;
                let before = raw(a.pResourceBefore.as_ref());
                let after = raw(a.pResourceAfter.as_ref());
                opaque |= before == 0 || after == 0;
                [before, after]
            }
            D3D12_RESOURCE_BARRIER_TYPE_UAV => [raw(b.Anonymous.UAV.pResource.as_ref()), 0],
            _ => [0, 0],
        }
    });
    note_list_resources(list, resources);
    if opaque {
        note_list_untracked(list);
    }
}

#[cfg(test)]
#[test]
fn native_timestamp_resolve_does_not_require_adoption() {
    unsafe extern "system" fn resolve(
        _: Obj, _: Obj, kind: D3D12_QUERY_TYPE, first: u32, count: u32, dst: Obj, offset: u64,
    ) {
        dst.cast::<(D3D12_QUERY_TYPE, u32, u32, u64)>().write((kind, first, count, offset));
    }

    let slot = std::mem::offset_of!(ID3D12GraphicsCommandList_Vtbl, ResolveQueryData) / size_of::<usize>();
    let mut table = [0usize; 60];
    table[slot] = resolve as usize;
    let mut list = table.as_ptr();
    let mut received = (D3D12_QUERY_TYPE(0), 0u32, 0u32, 0u64);
    unsafe {
        ResolveQueryData(
            (&raw mut list).cast(), std::ptr::null_mut(), D3D12_QUERY_TYPE_TIMESTAMP,
            3, 2, (&raw mut received).cast(), 64,
        );
    }
    assert_eq!(received, (D3D12_QUERY_TYPE_TIMESTAMP, 3, 2, 64));
}
