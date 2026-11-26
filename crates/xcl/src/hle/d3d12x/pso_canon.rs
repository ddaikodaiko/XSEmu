//! One byte form for each pipeline description: the title fills the padding of its pipeline
//! state streams with whatever its memory held (a depth-stencil's two bytes after the stencil
//! masks, three after each blend target's write mask, an input layout's last four), and leaves
//! state the GPU ignores as it was (blend factors with blending off, targets past the count). A
//! pipeline kept by those bytes (`pso_manifest`, the driver blob names) was kept again under
//! a new key each run. Here every byte no field covers is zeroed and every ignored field set to
//! D3D12's default, so one pipeline has one form: before it is keyed and before the driver sees it.
//! `XCL_PSO_CANONICAL=0`: the bytes are kept as the title wrote them (records say which).

use std::mem::{offset_of, size_of};
use windows::Win32::Graphics::Direct3D12::*;

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PSO_CANONICAL").map_or(true, |v| v != "0"))
}

fn field_size<T, F>(_: fn(&T) -> &F) -> usize {
    size_of::<F>()
}

/// A struct's size and its fields' (offset, size), in order.
macro_rules! layout {
    ($t:ty: $($f:ident),+) => {
        (size_of::<$t>(), [$((offset_of!($t, $f), field_size(|t: &$t| &t.$f))),+])
    };
}

/// Zero the bytes of the struct at `at` that none of `fields` covers.
fn gaps(p: &mut [u8], at: usize, (size, fields): (usize, impl AsRef<[(usize, usize)]>)) {
    let mut end = 0;
    for &(o, n) in fields.as_ref() {
        p[at + end..at + o].fill(0);
        end = o + n;
    }
    p[at + end..at + size].fill(0);
}

fn word(p: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(p[at..at + 4].try_into().unwrap())
}

fn set(p: &mut [u8], at: usize, v: i32) {
    p[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn render_target(p: &mut [u8], at: usize, used: bool) {
    type T = D3D12_RENDER_TARGET_BLEND_DESC;
    if !used {
        p[at..at + size_of::<T>()].fill(0);
        p[at + offset_of!(T, RenderTargetWriteMask)] = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
    }
    if !used || word(p, at + offset_of!(T, BlendEnable)) == 0 {
        set(p, at + offset_of!(T, SrcBlend), D3D12_BLEND_ONE.0);
        set(p, at + offset_of!(T, DestBlend), D3D12_BLEND_ZERO.0);
        set(p, at + offset_of!(T, BlendOp), D3D12_BLEND_OP_ADD.0);
        set(p, at + offset_of!(T, SrcBlendAlpha), D3D12_BLEND_ONE.0);
        set(p, at + offset_of!(T, DestBlendAlpha), D3D12_BLEND_ZERO.0);
        set(p, at + offset_of!(T, BlendOpAlpha), D3D12_BLEND_OP_ADD.0);
    }
    if !used || word(p, at + offset_of!(T, LogicOpEnable)) == 0 {
        set(p, at + offset_of!(T, LogicOp), D3D12_LOGIC_OP_NOOP.0);
    }
    gaps(p, at, layout!(T: BlendEnable, LogicOpEnable, SrcBlend, DestBlend, BlendOp, SrcBlendAlpha, DestBlendAlpha, BlendOpAlpha, LogicOp, RenderTargetWriteMask));
}

/// `targets`: the pipeline's render target count, when known.
fn blend(p: &mut [u8], at: usize, targets: Option<usize>) {
    type B = D3D12_BLEND_DESC;
    let independent = word(p, at + offset_of!(B, IndependentBlendEnable)) != 0;
    let used = if independent { targets.unwrap_or(8) } else { 1 };
    for i in 0..8 {
        render_target(p, at + offset_of!(B, RenderTarget) + i * size_of::<D3D12_RENDER_TARGET_BLEND_DESC>(), i < used);
    }
    gaps(p, at, layout!(B: AlphaToCoverageEnable, IndependentBlendEnable, RenderTarget));
}

fn stencil_face(p: &mut [u8], at: usize) {
    type F = D3D12_DEPTH_STENCILOP_DESC;
    set(p, at + offset_of!(F, StencilFailOp), D3D12_STENCIL_OP_KEEP.0);
    set(p, at + offset_of!(F, StencilDepthFailOp), D3D12_STENCIL_OP_KEEP.0);
    set(p, at + offset_of!(F, StencilPassOp), D3D12_STENCIL_OP_KEEP.0);
    set(p, at + offset_of!(F, StencilFunc), D3D12_COMPARISON_FUNC_ALWAYS.0);
}

/// The stencil state of a depth-stencil state of any of the three versions (they agree up to the
/// faces) when stencil is off: `masks` the offsets of its stencil read / write masks, `faces` of
/// its two faces. (The depth write mask and function are kept as they are with depth off.)
fn stencil(p: &mut [u8], at: usize, masks: &[usize], faces: [usize; 2]) {
    if word(p, at + offset_of!(D3D12_DEPTH_STENCIL_DESC, StencilEnable)) == 0 {
        for &m in masks {
            p[at + m] = 0xFF;
        }
        for f in faces {
            stencil_face(p, at + f);
        }
    }
}

/// Subobject payload `p` of type `ty` (`null::subobject_payload`'s table) in its one form.
pub fn payload(ty: u32, p: &mut [u8], targets: Option<usize>) {
    match ty {
        7 => gaps(p, 0, layout!(D3D12_STREAM_OUTPUT_DESC: pSODeclaration, NumEntries, pBufferStrides, NumStrides, RasterizedStream)),
        8 => blend(p, 0, targets),
        10 | 27 => gaps(p, 0, layout!(D3D12_RASTERIZER_DESC: FillMode, CullMode, FrontCounterClockwise, DepthBias, DepthBiasClamp, SlopeScaledDepthBias, DepthClipEnable, MultisampleEnable, AntialiasedLineEnable, ForcedSampleCount, ConservativeRaster)),
        28 => gaps(p, 0, layout!(D3D12_RASTERIZER_DESC2: FillMode, CullMode, FrontCounterClockwise, DepthBias, DepthBiasClamp, SlopeScaledDepthBias, DepthClipEnable, LineRasterizationMode, ForcedSampleCount, ConservativeRaster)),
        11 => {
            type D = D3D12_DEPTH_STENCIL_DESC;
            stencil(p, 0, &[offset_of!(D, StencilReadMask), offset_of!(D, StencilWriteMask)], [offset_of!(D, FrontFace), offset_of!(D, BackFace)]);
            gaps(p, 0, layout!(D: DepthEnable, DepthWriteMask, DepthFunc, StencilEnable, StencilReadMask, StencilWriteMask, FrontFace, BackFace));
        }
        21 => {
            type D = D3D12_DEPTH_STENCIL_DESC1;
            stencil(p, 0, &[offset_of!(D, StencilReadMask), offset_of!(D, StencilWriteMask)], [offset_of!(D, FrontFace), offset_of!(D, BackFace)]);
            gaps(p, 0, layout!(D: DepthEnable, DepthWriteMask, DepthFunc, StencilEnable, StencilReadMask, StencilWriteMask, FrontFace, BackFace, DepthBoundsTestEnable));
        }
        26 => {
            type D = D3D12_DEPTH_STENCIL_DESC2;
            type F = D3D12_DEPTH_STENCILOP_DESC1;
            let (front, back) = (offset_of!(D, FrontFace), offset_of!(D, BackFace));
            let (read, write) = (offset_of!(F, StencilReadMask), offset_of!(F, StencilWriteMask));
            stencil(p, 0, &[front + read, front + write, back + read, back + write], [front, back]);
            for f in [front, back] {
                gaps(p, f, layout!(F: StencilFailOp, StencilDepthFailOp, StencilPassOp, StencilFunc, StencilReadMask, StencilWriteMask));
            }
            gaps(p, 0, layout!(D: DepthEnable, DepthWriteMask, DepthFunc, StencilEnable, FrontFace, BackFace, DepthBoundsTestEnable));
        }
        12 => gaps(p, 0, layout!(D3D12_INPUT_LAYOUT_DESC: pInputElementDescs, NumElements)),
        15 => {
            let n = (word(p, offset_of!(D3D12_RT_FORMAT_ARRAY, NumRenderTargets)) as usize).min(8);
            p[offset_of!(D3D12_RT_FORMAT_ARRAY, RTFormats) + 4 * n..offset_of!(D3D12_RT_FORMAT_ARRAY, NumRenderTargets)].fill(0);
        }
        22 => gaps(p, 0, layout!(D3D12_VIEW_INSTANCING_DESC: ViewInstanceCount, pViewInstanceLocations, Flags)),
        _ => {}
    }
}

/// Pipeline state stream `buf` in its one form, in place (the subobjects up to the first of a
/// type not known; each type's padding before and after its payload too).
pub fn stream(buf: &mut [u8]) {
    let mut subs = Vec::new();
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = word(buf, off);
        let Some((size, align)) = super::null::subobject_payload(ty) else { break };
        let at = off + if align == 8 { 8 } else { 4 };
        if at + size > buf.len() {
            break;
        }
        let next = (off + ((at - off + size + 7) & !7)).min(buf.len());
        subs.push((ty, off, at, size, next));
        off = next;
    }
    let targets = subs.iter().find(|s| s.0 == 15).map(|s| (word(buf, s.2 + offset_of!(D3D12_RT_FORMAT_ARRAY, NumRenderTargets)) as usize).min(8));
    for (ty, off, at, size, next) in subs {
        buf[off + 4..at].fill(0);
        payload(ty, &mut buf[at..at + size], targets);
        buf[at + size..next].fill(0);
    }
}

/// The bytes of a `D3D12_COMPUTE_PIPELINE_STATE_DESC` in their one form.
pub fn compute(d: &mut [u8]) {
    gaps(d, 0, layout!(D3D12_COMPUTE_PIPELINE_STATE_DESC: pRootSignature, CS, NodeMask, CachedPSO, Flags));
}

/// The bytes of a `D3D12_GRAPHICS_PIPELINE_STATE_DESC` in their one form.
pub fn graphics(d: &mut [u8]) {
    type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
    let targets = (word(d, offset_of!(G, NumRenderTargets)) as usize).min(8);
    let state = [(7, offset_of!(G, StreamOutput), size_of::<D3D12_STREAM_OUTPUT_DESC>()), (8, offset_of!(G, BlendState), size_of::<D3D12_BLEND_DESC>()), (10, offset_of!(G, RasterizerState), size_of::<D3D12_RASTERIZER_DESC>()), (11, offset_of!(G, DepthStencilState), size_of::<D3D12_DEPTH_STENCIL_DESC>()), (12, offset_of!(G, InputLayout), size_of::<D3D12_INPUT_LAYOUT_DESC>())];
    for (ty, at, size) in state {
        payload(ty, &mut d[at..at + size], Some(targets));
    }
    d[offset_of!(G, RTVFormats) + 4 * targets..offset_of!(G, RTVFormats) + 32].fill(0);
    gaps(d, 0, layout!(G: pRootSignature, VS, PS, DS, HS, GS, StreamOutput, BlendState, SampleMask, RasterizerState, DepthStencilState, InputLayout, IBStripCutValue, PrimitiveTopologyType, NumRenderTargets, RTVFormats, DSVFormat, SampleDesc, NodeMask, CachedPSO, Flags));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream of `subs` with `fill` in every byte between and after the payloads.
    fn stream_of(subs: &[(u32, Vec<u8>)], fill: u8) -> Vec<u8> {
        let mut out = Vec::new();
        for (ty, p) in subs {
            let (size, align) = super::super::null::subobject_payload(*ty).unwrap();
            assert_eq!(size, p.len());
            out.extend_from_slice(&ty.to_le_bytes());
            out.resize(out.len() + if align == 8 { 4 } else { 0 }, fill);
            out.extend_from_slice(p);
            out.resize((out.len() + 7) & !7, fill);
        }
        out
    }

    fn put(p: &mut [u8], at: usize, v: u32) {
        p[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// A title's stream: one target, blending on its first, depth on, stencil off, an input
    /// layout; `fill` in its padding and in the state the GPU ignores.
    fn title_stream(fill: u8) -> Vec<u8> {
        let mut rt = vec![fill; size_of::<D3D12_RT_FORMAT_ARRAY>()];
        put(&mut rt, 0, 28);
        put(&mut rt, 32, 1);
        let mut blend = vec![fill; size_of::<D3D12_BLEND_DESC>()];
        put(&mut blend, 0, 0);
        put(&mut blend, 4, 1);
        for (at, v) in [(0, 1), (4, 0), (8, 5), (12, 6), (16, 1), (20, 2), (24, 1), (28, 1)] {
            put(&mut blend, 8 + at, v);
        }
        blend[8 + 36] = 15;
        let mut depth = vec![fill; size_of::<D3D12_DEPTH_STENCIL_DESC1>()];
        for (at, v) in [(0, 1), (4, 1), (8, 4), (12, 0), (52, 0)] {
            put(&mut depth, at, v);
        }
        let mut layout = vec![fill; size_of::<D3D12_INPUT_LAYOUT_DESC>()];
        layout[..8].copy_from_slice(&0x1000u64.to_le_bytes());
        put(&mut layout, 8, 3);
        stream_of(&[(15, rt), (8, blend), (21, depth), (12, layout), (14, 3u32.to_le_bytes().to_vec())], fill)
    }

    fn key(desc: Vec<u8>) -> u64 {
        super::super::pso_manifest::Record { kind: super::super::pso_manifest::Kind::MeshStream, desc, shaders: Vec::new(), layout: None, canonical: true }.key()
    }

    fn canonical(mut s: Vec<u8>) -> Vec<u8> {
        stream(&mut s);
        s
    }

    #[test]
    fn padding_and_ignored_state_give_one_stream_and_one_key() {
        let (a, b) = (title_stream(0x00), title_stream(0xA7));
        assert_ne!(a, b);
        assert_ne!(key(a.clone()), key(b.clone()));
        let (a, b) = (canonical(a), canonical(b));
        assert_eq!(a, b);
        assert_eq!(key(a.clone()), key(b));
        assert_eq!(canonical(a.clone()), a);
    }

    #[test]
    fn the_padding_titles_leave_is_zeroed() {
        let s = canonical(title_stream(0xA7));
        let at = |ty: u32| {
            let mut off = 0;
            loop {
                let t = u32::from_le_bytes(s[off..off + 4].try_into().unwrap());
                let (size, align) = super::super::null::subobject_payload(t).unwrap();
                let p = off + if align == 8 { 8 } else { 4 };
                if t == ty {
                    return p;
                }
                off += (p - off + size + 7) & !7;
            }
        };
        assert_eq!(s[at(21) + 18..at(21) + 20], [0, 0]);
        assert_eq!(s[at(8) + 8 + 37..at(8) + 8 + 40], [0, 0, 0]);
        assert_eq!(s[at(12) + 12..at(12) + 16], [0; 4]);
        assert_eq!(s[at(12) - 4..at(12)], [0; 4]);
    }

    #[test]
    fn real_state_differences_keep_their_own_keys() {
        let base = canonical(title_stream(0xA7));
        let changed = |at: usize, v: u32| {
            let mut s = title_stream(0xA7);
            put(&mut s, at, v);
            canonical(s)
        };
        let blend = 40 + 4;
        let depth = blend + 336;
        for s in [changed(blend + 8 + 8, 4), changed(blend + 8 + 36, 7), changed(depth + 8, 2), changed(depth + 12, 1), changed(4 + 32, 2)] {
            assert_ne!(s, base);
            assert_ne!(key(s), key(base.clone()));
        }
    }

    #[test]
    fn every_subobject_type_the_table_knows_has_its_struct_size() {
        for (ty, size) in [
            (7, size_of::<D3D12_STREAM_OUTPUT_DESC>()),
            (8, size_of::<D3D12_BLEND_DESC>()),
            (10, size_of::<D3D12_RASTERIZER_DESC>()),
            (27, size_of::<D3D12_RASTERIZER_DESC1>()),
            (28, size_of::<D3D12_RASTERIZER_DESC2>()),
            (11, size_of::<D3D12_DEPTH_STENCIL_DESC>()),
            (21, size_of::<D3D12_DEPTH_STENCIL_DESC1>()),
            (26, size_of::<D3D12_DEPTH_STENCIL_DESC2>()),
            (12, size_of::<D3D12_INPUT_LAYOUT_DESC>()),
            (15, size_of::<D3D12_RT_FORMAT_ARRAY>()),
            (22, size_of::<D3D12_VIEW_INSTANCING_DESC>()),
            (17, size_of::<windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC>()),
            (19, size_of::<D3D12_CACHED_PIPELINE_STATE>()),
        ] {
            assert_eq!(super::super::null::subobject_payload(ty).map(|p| p.0), Some(size), "subobject {ty}");
        }
        for ty in 0..32 {
            if let Some((size, _)) = super::super::null::subobject_payload(ty) {
                let mut p = vec![0xA7; size];
                payload(ty, &mut p, Some(1));
                let once = p.clone();
                payload(ty, &mut p, Some(1));
                assert_eq!(p, once, "subobject {ty}");
            }
        }
    }

    #[test]
    fn descriptions_lose_their_padding() {
        type C = D3D12_COMPUTE_PIPELINE_STATE_DESC;
        let mut c = vec![0xA7; size_of::<C>()];
        compute(&mut c);
        assert_eq!(c[offset_of!(C, NodeMask) + 4..offset_of!(C, CachedPSO)], [0; 4]);
        assert_eq!(c[offset_of!(C, Flags) + 4..], [0; 4]);
        type G = D3D12_GRAPHICS_PIPELINE_STATE_DESC;
        let mut g = G::default();
        g.NumRenderTargets = 1;
        g.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28);
        g.SampleMask = u32::MAX;
        g.DepthStencilState.DepthEnable = true.into();
        g.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        g.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER_EQUAL;
        g.BlendState.RenderTarget[0].RenderTargetWriteMask = 15;
        let mut a = unsafe { std::slice::from_raw_parts(&g as *const G as *const u8, size_of::<G>()) }.to_vec();
        let mut b = a.clone();
        let ds = offset_of!(G, DepthStencilState);
        let rt = offset_of!(G, BlendState) + offset_of!(D3D12_BLEND_DESC, RenderTarget);
        for r in [ds + size_of::<D3D12_DEPTH_STENCIL_DESC>()..offset_of!(G, InputLayout), offset_of!(G, Flags) + 4..size_of::<G>(), ds + 18..ds + 20, rt + 37..rt + 40, rt + 40..rt + 80, offset_of!(G, RTVFormats) + 4..offset_of!(G, RTVFormats) + 8, rt + 8..rt + 32] {
            b[r].fill(0xA7);
        }
        assert_ne!(a, b);
        graphics(&mut a);
        graphics(&mut b);
        assert_eq!(a, b);
    }
}
