//! The memory-path and SDWA helpers written without short-circuit operators (`and`, `or`,
//! `select` instead of `&&`, `||`, `?:`, which HLSL 2021 makes branches) against their
//! earlier text: both are compiled by the direct backend (`xcl_dxil`) and run by its
//! interpreter on operands that land on the edges of every test (null high words, the UMA
//! base, the last window, the 64 KB page and 32-bit address ends), and must store the same.
//! A changed predicate must show (the harness is not blind to the path).

use xcl_dxil::interp::{self, Config, Verdict};

const OLD_PAGE_DIRECTORY: &str = r#"
static const uint XCL_PTE_ABSENT = 0u, XCL_PTE_UNMAPPED = 1u, XCL_PTE_MAPPED = 2u, XCL_PTE_INVALID = 3u;
uint xcl_leaf_pte(uint srv, uint lo) {
    if (srv == 0u) return 0u;
    ByteAddressBuffer leaf = ResourceDescriptorHeap[NonUniformResourceIndex(srv)];
    return leaf.Load((lo >> 16) * 4u);
}
uint2 xcl_lookup_gpu_pte(uint2 va, uint directory) {
    if (va.y > 65535u) return uint2(XCL_PTE_INVALID, 0u);
    if (directory == 0u) return uint2(XCL_PTE_ABSENT, 0u);
    ByteAddressBuffer root = ResourceDescriptorHeap[directory];
    uint2 leaves = root.Load2(va.y * 8u);
    uint pte = xcl_leaf_pte(leaves.x, va.x);
    if (pte == 0u) pte = xcl_leaf_pte(leaves.y, va.x);
    if (pte == 0u) return uint2(XCL_PTE_ABSENT, 0u);
    if (pte == 0x80000000u) return uint2(XCL_PTE_UNMAPPED, pte);
    if ((pte & 0xf8000000u) != 0x80000000u || (pte & 0xfffffu) == 0u) return uint2(XCL_PTE_INVALID, pte);
    return uint2(XCL_PTE_MAPPED, pte);
}
uint2 xcl_raw_gpuva(uint2 va, uint directory, uint mem_views, uint uma_hi, uint windows) {
    uint2 miss = uint2(mem_views + 4u, 0u);
    uint2 p = xcl_lookup_gpu_pte(va, directory);
    if (p.x == XCL_PTE_MAPPED) return uint2(p.y & 0xfffffu, ((p.y >> 20) & 127u) * 65536u + (va.x & 65535u));
    if (p.x != XCL_PTE_ABSENT) return miss;
    uint w = va.y - uma_hi;
    return w < windows ? uint2(mem_views + w, va.x) : miss;
}
"#;
const OLD_PHYSICAL_ACCESS: &str = r#"
static uint xcl_page_table = 0u;
static uint64_t xcl_ea = 0;
static uint xcl_origin = 0u;

uint xcl_topology_for_recording(uint locator) {
    if (locator == 0u) return 0u;
    ByteAddressBuffer selected = ResourceDescriptorHeap[locator];
    return selected.Load(0u);
}

uint64_t xcl_operand_va(uint a) {
    if (xcl_ea == 0 && xcl_origin == 0u) return (uint64_t)a;
    return xcl_ea + (uint64_t)(int64_t)(int)(a - (uint)xcl_ea);
}
bool xcl_direct(uint64_t va, uint w, uint bytes) {
    uint hi = (uint)(va >> 32);
    return xcl_origin <= 65535u && w < xcl_windows && (uint)va <= 0xffffffffu - (bytes - 1u)
        && (xcl_origin == 0u ? hi == 0u && w == 0u : hi == xcl_uma_base + w);
}
uint2 xcl_resolve_va(uint64_t va) {
    uint hi = (uint)(va >> 32), lo = (uint)va;
    uint2 miss = uint2(xcl_mem_views + 4u, 0u);
    if (hi > 65535u || xcl_origin > 65535u) return miss;
    if ((xcl_origin == 0u) != (hi == 0u)) return miss;
    return xcl_raw_gpuva(uint2(lo, hi == 0u ? xcl_uma_base : hi), xcl_page_table, xcl_mem_views, xcl_uma_base, xcl_windows);
}
uint2 xcl_resolve_read(uint a) { return xcl_resolve_va(xcl_operand_va(a)); }
"#;
const OLD_SDWA: &str = r#"uint sdwa_sel(uint x, uint sel, bool sext) {
    if (sel >= 6u) return x;
    uint sh = sel < 4u ? 8u * sel : 16u * (sel - 4u); uint w = sel < 4u ? 8u : 16u;
    uint v = (x >> sh) & ((1u << w) - 1u);
    return sext ? (uint)(((int)(v << (32u - w))) >> (32u - w)) : v;
}
uint sdwa_dst(uint r, uint old, uint sel, uint unused) {
    if (sel >= 6u) return r;
    uint sh = sel < 4u ? 8u * sel : 16u * (sel - 4u); uint w = sel < 4u ? 8u : 16u;
    uint m = ((1u << w) - 1u) << sh; uint v = (r << sh) & m;
    if (unused == 2u) return (old & ~m) | v;
    if (unused == 1u) { uint top = sh + w; bool s = ((r >> (w - 1u)) & 1u) != 0u; return v | ((s && top < 32u) ? (0xFFFFFFFFu << top) : 0u); }
    return v;
}
"#;

const OLD_ORIGIN: &str = "void origin_flag(uint hi, uint64_t ea) { if (hi > 65535u || (hi == 0u && (uint)(ea >> 32) != 0u)) xcl_origin = 0xffffffffu; }
void align_flag(uint m) { if (((uint)xcl_ea & m) != 0u) xcl_origin = 0xffffffffu; }";
const NEW_ORIGIN: &str = "void origin_flag(uint hi, uint64_t ea) { xcl_origin = select(or(hi > 65535u, and(hi == 0u, (uint)(ea >> 32) != 0u)), 0xffffffffu, xcl_origin); }
void align_flag(uint m) { xcl_origin = select(((uint)xcl_ea & m) != 0u, 0xffffffffu, xcl_origin); }";

const DRIVER: &str = r#"
ByteAddressBuffer xin : register(t0);
RWByteAddressBuffer xout : register(u0);
uint edge(uint x, uint k) {
    uint s = (k >> 28) & 7u;
    return select(s == 0u, 0u, select(s == 1u, x & 15u, select(s == 2u, 65534u + (x & 3u), select(s == 3u, xcl_uma_base + (x & 7u),
        select(s == 4u, 0xfffffff0u + (x & 15u), select(s == 5u, 1u, x))))));
}
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    xcl_uma_base = 16u + (a.x & 3u);
    xcl_windows = 1u + ((a.x >> 2) & 3u);
    xcl_mem_views = 100u;
    xcl_page_table = select((a.y & 3u) == 0u, a.y, 0u);
    xcl_origin = edge(a.z, b.x);
    uint lo = edge(b.z, a.w);
    xcl_ea = select((b.x & 7u) == 0u, 0ull, ((uint64_t)edge(b.w, a.y) << 32) | (uint64_t)lo);
    uint hi = edge(a.w, b.y);
    uint w = edge(b.x ^ a.w, b.z) & 7u;
    uint64_t va = xcl_operand_va(edge(b.z ^ a.x, a.z));
    uint d = (uint)xcl_direct(va, w, 1u) | ((uint)xcl_direct(va, w, 4u) << 1) | ((uint)xcl_direct(va, w, 8u) << 2) | ((uint)xcl_direct(va, w, 16u) << 3);
    uint2 r = xcl_resolve_va(((uint64_t)hi << 32) | (uint64_t)lo);
    uint2 rr = xcl_resolve_read(lo ^ b.y);
    origin_flag(hi, xcl_ea);
    uint o1 = xcl_origin;
    align_flag(select((a.x & 1u) != 0u, 7u, 3u));
    uint sel = sdwa_sel(a.x, b.y % 7u, (b.z & 1u) != 0u);
    uint dst = sdwa_dst(a.y, b.x, b.w % 7u, a.w % 3u);
    xout.Store4(i * 48u, uint4((uint)va, (uint)(va >> 32), d, r.x));
    xout.Store4(i * 48u + 16u, uint4(r.y, rr.x, rr.y, o1));
    xout.Store4(i * 48u + 32u, uint4(xcl_origin, sel, dst, 0u));
}
"#;

fn program(dir: &str, phys: &str, sdwa: &str, origin: &str) -> interp::Program {
    let src = format!("{dir}\nstatic uint xcl_mem_views = 0u, xcl_uma_base = 0u, xcl_windows = 0u;\n{phys}\n{sdwa}\n{origin}\n{DRIVER}");
    interp::load(&xcl_dxil::compile(&src, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{src}"))).unwrap()
}

fn new_sdwa() -> &'static str {
    let p = super::emit::PRELUDE;
    let i = p.find("uint sdwa_sel(").unwrap();
    let j = p[p.find("uint sdwa_dst(").unwrap()..].find("\n}\n").unwrap() + p.find("uint sdwa_dst(").unwrap() + 3;
    &p[i..j]
}

fn verdict(a: &interp::Program, b: &interp::Program) -> Verdict {
    interp::diff(a, b, 0..24, &Config::default()).0
}

#[test]
fn branch_free_memory_and_sdwa_helpers_compute_the_same() {
    let old = program(OLD_PAGE_DIRECTORY, OLD_PHYSICAL_ACCESS, OLD_SDWA, OLD_ORIGIN);
    let new = program(super::GPU_PAGE_DIRECTORY_HLSL, super::emit::PHYSICAL_ACCESS, new_sdwa(), NEW_ORIGIN);
    assert_eq!(verdict(&old, &new), Verdict::Equal);
    assert!(new_sdwa().contains("select(") && !new_sdwa().contains(" ? "));
    for (text, from, to) in [
        (super::emit::PHYSICAL_ACCESS, "select(xcl_origin == 0u, and(hi == 0u, w == 0u)", "select(xcl_origin != 0u, and(hi == 0u, w == 0u)"),
        (super::emit::PHYSICAL_ACCESS, "or(or(hi > 65535u, xcl_origin > 65535u)", "or(and(hi > 65535u, xcl_origin > 65535u)"),
        (super::emit::PHYSICAL_ACCESS, "select(and(xcl_ea == 0, xcl_origin == 0u), (uint64_t)a,", "select(or(xcl_ea == 0, xcl_origin == 0u), (uint64_t)a,"),
        (super::GPU_PAGE_DIRECTORY_HLSL, "select(w < windows, mem_views + w, miss.x)", "select(w <= windows, mem_views + w, miss.x)"),
        (NEW_ORIGIN, "select(or(hi > 65535u, and(", "select(and(hi > 65535u, or("),
    ] {
        assert!(text.contains(from), "{from}");
        let m = text.replace(from, to);
        let (dir, phys, origin) = match () {
            _ if std::ptr::eq(text, super::GPU_PAGE_DIRECTORY_HLSL) => (m.as_str(), super::emit::PHYSICAL_ACCESS, NEW_ORIGIN),
            _ if std::ptr::eq(text, NEW_ORIGIN) => (super::GPU_PAGE_DIRECTORY_HLSL, super::emit::PHYSICAL_ACCESS, m.as_str()),
            _ => (super::GPU_PAGE_DIRECTORY_HLSL, m.as_str(), NEW_ORIGIN),
        };
        let mutant = program(dir, phys, new_sdwa(), origin);
        assert!(matches!(verdict(&old, &mutant), Verdict::Different(_)), "mutant {to} not caught");
    }
}

const OLD_RLA: &str = r#"uint rla(uint v, uint l) {
    uint target = l + 32u * xcl_half;
    bool in_range = target < WaveGetLaneCount();
    uint lane = WaveGetLaneIndex();
    uint h = select(in_range, target, 0u);
    uint w = WaveReadLaneAt(v, h);
#ifdef XCL_PIXEL
    // A lane of the reader's own quad through a quad read: those see helper lanes (wave reads
    // do not before SM 6.7), and the console's derivatives (DPP quad_perm, ds_swizzle) read the
    // helper pixels of a quad a primitive's edge splits.
    uint q0 = QuadReadLaneAt(v, 0u), q1 = QuadReadLaneAt(v, 1u), q2 = QuadReadLaneAt(v, 2u), q3 = QuadReadLaneAt(v, 3u);
    uint q = (l & 2u) != 0u ? ((l & 1u) != 0u ? q3 : q2) : ((l & 1u) != 0u ? q1 : q0);
    // A helper lane of another quad: after s_wqm the console's v_readfirstlane often picks
    // one (the first lane of EXEC on a primitive's edge), and a wave read of it is undefined
    // here (0 in practice: every other quad of the wave got the wrong value, e.g. a shader's
    // draw type, and took the wrong branch). Its own quad sees it through a quad read; the
    // reader takes that from a live lane of the same quad.
    uint4 xcl_lv = WaveActiveBallot(true);
    uint nib = (xcl_lv[min(h >> 5, 3u)] >> (h & 28u)) & 15u;
    uint src = select(nib != 0u, (h & ~3u) + firstbitlow(nib), lane);
    uint r0 = WaveReadLaneAt(q0, src), r1 = WaveReadLaneAt(q1, src), r2 = WaveReadLaneAt(q2, src), r3 = WaveReadLaneAt(q3, src);
    uint r = (l & 2u) != 0u ? ((l & 1u) != 0u ? r3 : r2) : ((l & 1u) != 0u ? r1 : r0);
    if (nib != 0u && ((nib >> (h & 3u)) & 1u) == 0u) w = r;
    if ((l >> 2) == (lane_id >> 2)) w = q;
#endif
    return select(in_range, w, 0u);
}
"#;

const PS_DRIVER: &str = r#"
uint4 main(float4 pos : SV_Position, float4 c : COLOR0) : SV_Target {
    lane_id = WaveGetLaneIndex() & 31u;
    uint v = asuint(c.x) ^ (uint)pos.x, l = asuint(c.y);
    l = select((l & 1u) != 0u, (lane_id & ~3u) | ((l >> 1) & 3u), (l >> 1) & 31u);
    return uint4(rla(v, l), rla(v ^ 0x55u, lane_id ^ 1u), rla(v, (l + 2u) & 31u), l);
}
"#;

fn pixel_program(rla: &str) -> interp::Program {
    let src = format!("#define XCL_PIXEL 1\nstatic uint lane_id = 0u, xcl_half = 0u;\n{rla}\n{PS_DRIVER}");
    interp::load(&xcl_dxil::compile(&src, "main", "ps_6_6").unwrap_or_else(|e| panic!("{e}\n{src}"))).unwrap()
}

fn new_rla() -> &'static str {
    let p = super::emit::PRELUDE;
    let i = p.find("uint rla(uint v, uint l) {").unwrap();
    &p[i..i + p[i..].find("\n}\n").unwrap() + 3]
}

#[test]
fn branch_free_pixel_lane_read_computes_the_same() {
    let old = pixel_program(OLD_RLA);
    assert_eq!(verdict(&old, &pixel_program(new_rla())), Verdict::Equal);
    let from = "w = select((l >> 2) == (lane_id >> 2), q, w);";
    assert!(new_rla().contains(from) && !new_rla().contains(" ? "));
    let mutant = pixel_program(&new_rla().replace(from, "w = select((l >> 2) != (lane_id >> 2), q, w);"));
    assert!(matches!(verdict(&old, &mutant), Verdict::Different(_)));
}
