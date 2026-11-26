//! The memory accessors' path across a 64 KiB page (each dword its own direct test, else the
//! page it lies in, resolved once per page, a miss at offset 0 as before) against the earlier
//! text (each dword through the whole one-dword accessor): both are compiled by the direct backend (`xcl_dxil`) and run by its
//! interpreter on addresses at the page and 32-bit ends, with and without a page directory,
//! and must load and store the same. A wrong page, offset or direct test must show.

use xcl_dxil::interp::{self, Config, Verdict};

const OLD_ACCESSORS: &str = r#"
uint xcl_mLoad(uint a) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 4u))) { return xcl_mb.Load(a); } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; return b.Load(p.y); }
uint2 xcl_mLoad2(uint a) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 8u))) { return xcl_mb.Load2(a); } if ((a & 65535u) > 65528u) { return uint2(xcl_mLoad(a + 0u), xcl_mLoad(a + 4u)); } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; return b.Load2(p.y); }
uint3 xcl_mLoad3(uint a) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 12u))) { return xcl_mb.Load3(a); } if ((a & 65535u) > 65524u) { return uint3(xcl_mLoad(a + 0u), xcl_mLoad(a + 4u), xcl_mLoad(a + 8u)); } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; return b.Load3(p.y); }
uint4 xcl_mLoad4(uint a) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 16u))) { return xcl_mb.Load4(a); } if ((a & 65535u) > 65520u) { return uint4(xcl_mLoad(a + 0u), xcl_mLoad(a + 4u), xcl_mLoad(a + 8u), xcl_mLoad(a + 12u)); } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; return b.Load4(p.y); }
void xcl_mStore(uint a, uint v) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 4u))) { xcl_mb.Store(a, v); return; } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; b.Store(p.y, v); }
void xcl_mStore2(uint a, uint2 v) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 8u))) { xcl_mb.Store2(a, v); return; } if ((a & 65535u) > 65528u) { xcl_mStore(a + 0u, v[0]); xcl_mStore(a + 4u, v[1]); return; } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; b.Store2(p.y, v); }
void xcl_mStore3(uint a, uint3 v) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 12u))) { xcl_mb.Store3(a, v); return; } if ((a & 65535u) > 65524u) { xcl_mStore(a + 0u, v[0]); xcl_mStore(a + 4u, v[1]); xcl_mStore(a + 8u, v[2]); return; } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; b.Store3(p.y, v); }
void xcl_mStore4(uint a, uint4 v) { RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[xcl_mem_views + xcl_w]; uint64_t va = xcl_operand_va(a); if (and(xcl_page_table == 0u, xcl_direct(va, xcl_w, 16u))) { xcl_mb.Store4(a, v); return; } if ((a & 65535u) > 65520u) { xcl_mStore(a + 0u, v[0]); xcl_mStore(a + 4u, v[1]); xcl_mStore(a + 8u, v[2]); xcl_mStore(a + 12u, v[3]); return; } uint2 p = xcl_resolve_va(va); RWByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(p.x)]; b.Store4(p.y, v); }
"#;

const SETUP: &str = r#"
ByteAddressBuffer xin : register(t0);
RWByteAddressBuffer xout : register(u0);
uint edge(uint x, uint k) {
    uint s = (k >> 28) & 7u;
    return select(s == 0u, 0u, select(s == 1u, x & 15u, select(s == 2u, 65534u + (x & 3u), select(s == 3u, xcl_uma_base + (x & 7u),
        select(s == 4u, 0xfffffff0u + (x & 15u), select(s == 5u, 1u, x))))));
}
uint setup(uint4 a, uint4 b, uint c) {
    xcl_uma_base = select((a.x & 48u) == 0u, 0u, 16u + (a.x & 3u));
    xcl_windows = 1u + ((a.x >> 2) & 3u);
    xcl_mem_views = 100u;
    xcl_page_table = select((c & 3u) == 0u, 0u, (a.y & ~3u) | 4u);
    uint w = select((c & 4u) != 0u, (b.y >> 30) % xcl_windows, edge(b.x ^ a.w, b.z) & 7u);
    xcl_w = w;
    uint k = (c >> 3) & 3u;
    xcl_origin = select(k == 0u, 0u, select(k == 1u, xcl_uma_base + w, select(k == 2u, 0u, edge(a.z, b.x))));
    uint hi = select(k == 0u, 0u, select(k == 1u, xcl_uma_base + w + (b.w & 1u), select(k == 2u, b.w & 1u, edge(b.w, a.y))));
    uint s = (c >> 5) & 3u;
    uint lo = select(s == 0u, (b.z & 0xffff0000u) | (65520u + (b.z & 15u)), select(s == 1u, 0xfffffff0u + (b.z & 15u), select(s == 2u, 65532u + (b.z & 3u), b.z)));
    xcl_ea = select(and(k == 0u, (b.x & 1u) == 0u), 0ull, ((uint64_t)hi << 32) | (uint64_t)lo);
    return lo + select((a.z & 1u) == 0u, 0u, (a.z >> 1) & 12u) - select((a.z & 32u) == 0u, 0u, (a.z >> 6) & 12u);
}
"#;

const LOADS: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint p = setup(xin.Load4(i * 32u), xin.Load4(i * 32u + 16u), i ^ (i >> 6));
    uint4 l4 = xcl_mLoad4(p);
    uint3 l3 = xcl_mLoad3(p + 4u);
    uint2 l2 = xcl_mLoad2(p + 8u);
    xout.Store4(i * 48u, l4);
    xout.Store4(i * 48u + 16u, uint4(l3, xcl_mLoad(p)));
    xout.Store4(i * 48u + 32u, uint4(l2, p, 0u));
}
"#;

const STORES: &str = r#"
[numthreads(1, 1, 1)]
void main() {
    for (uint k = 0u; k < 64u; k++) {
        uint4 a = xin.Load4(k * 32u), b = xin.Load4(k * 32u + 16u);
        uint p = setup(a, b, k * 5u + (a.w >> 25));
        xcl_mStore4(p, a ^ b);
        xcl_mStore3(p + 4u + 4u * (k & 1u), b.xyz);
        xcl_mStore2(p + 8u, a.zw);
        xcl_mStore(p + 12u, k);
    }
}
"#;

fn new_accessors() -> String {
    let all = super::emit::plain_mem_accessors(4, false);
    let names = ["uint xcl_mLoad(", "uint2 xcl_mLoad2(", "uint3 xcl_mLoad3(", "uint4 xcl_mLoad4(", "void xcl_mStore(", "void xcl_mStore2(", "void xcl_mStore3(", "void xcl_mStore4("];
    all.lines().filter(|l| names.iter().any(|n| l.starts_with(n))).map(|l| format!("{l}\n")).collect()
}

fn program(accessors: &str, driver: &str) -> interp::Program {
    let src = format!(
        "static uint xcl_mem_views = 0u, xcl_uma_base = 0u, xcl_windows = 0u, xcl_w = 0u;\n{}\n{}\n{SETUP}\n{accessors}\n{driver}",
        super::GPU_PAGE_DIRECTORY_HLSL,
        super::emit::PHYSICAL_ACCESS
    );
    interp::load(&xcl_dxil::compile(&src, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{src}"))).unwrap()
}

fn verdicts(accessors: &str) -> [Verdict; 2] {
    [LOADS, STORES].map(|d| interp::diff(&program(OLD_ACCESSORS, d), &program(accessors, d), 0..24, &Config::default()).0)
}

#[test]
fn page_crossing_accessors_resolve_each_page_once_and_compute_the_same() {
    let new = new_accessors();
    assert_eq!(new.lines().count(), 8);
    assert!(new.contains("uint2 q = xcl_resolve_va(") && !new.contains("xcl_mLoad(a + 4u)") && !new.contains("xcl_mStore(a + 4u"));
    assert_eq!(verdicts(&new), [Verdict::Equal, Verdict::Equal]);
}

#[test]
fn a_wrong_page_offset_or_direct_test_is_caught() {
    let new = new_accessors();
    for (from, to) in [
        ("(a | 65535u) + 2u", "(a | 65535u)"),
        ("p.y + 4u)", "p.y + 0u)"),
        ("((a2 & 65535u) - 1u)", "((a1 & 65535u) - 1u)"),
        ("select(p.y == 0u, 0u, p.y + 8u)", "(p.y + 8u)"),
        ("xcl_direct(xcl_operand_va(a1), xcl_w, 4u)", "xcl_direct(xcl_operand_va(a0), xcl_w, 4u)"),
        ("bool h3 = (a3 >> 16) == (a >> 16)", "bool h3 = (a3 >> 16) == (a2 >> 16)"),
    ] {
        assert!(new.contains(from), "{from}");
        let v = verdicts(&new.replace(from, to));
        assert!(v.iter().any(|v| matches!(v, Verdict::Different(_))), "mutant {to} not caught: {v:?}");
    }
}
