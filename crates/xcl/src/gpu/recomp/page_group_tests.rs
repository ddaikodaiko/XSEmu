//! Page groups (`Em::page_group`): loads at constant offsets from one base read their pages from
//! two resolutions made once for the group (`xcl_group_pages`, `xcl_group_at`) instead of resolving
//! each, against the accessors of each load alone. Both are compiled by the direct backend
//! (`xcl_dxil`) and run by its interpreter: the accessors on addresses at the page and 32-bit
//! ends, with and without a page directory, at every offset a group admits, with lanes whose
//! result the load drops (address 0); and whole translated programs with the groups on and off.
//! As in every translated load, a load's address is the low word of its effective address
//! (`xcl_ea`, aligned down to 4 bytes) and the loads of a group are one effective address plus
//! their offsets.
//! A wrong page, offset or miss must show.

use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;
use xcl_dxil::interp::{self, Config, Verdict};

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
    uint lo = select(s == 0u, (b.z & 0xffff0000u) | (65520u + (b.z & 15u)), select(s == 1u, 0xfffffff0u + (b.z & 15u), select(s == 2u, (b.z & 0xffff0000u) | (b.z & 3u), b.z)));
    xcl_ea = select(and(k == 0u, (b.x & 1u) == 0u), 0ull, ((uint64_t)hi << 32) | (uint64_t)lo);
    return (uint)xcl_ea & ~3u;
}
"#;

const DRIVER: &str = r#"
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint4 a = xin.Load4(i * 32u), b = xin.Load4(i * 32u + 16u);
    uint p = setup(a, b, i ^ (i >> 7));
    uint64_t e0 = xcl_ea;
    uint64_t g = xcl_operand_va(p);
    uint4 r = xcl_group_pages(g);
    uint m = a.w ^ b.y;
    uint o = i * 512u;
    [unroll] for (uint k = 0u; k < 8u; k++) {
        uint d = k == 7u ? 65504u : k == 6u ? 4096u : k == 5u ? 65280u : 4u * k * (k + 1u);
        xcl_ea = e0 + d;
        bool ok = ((m >> k) & 1u) != 0u;
        bool4 q = bool4(((m >> (8u + k)) & 1u) != 0u, ((m >> (16u + k)) & 1u) != 0u, ((m >> (24u + k)) & 1u) == 0u, true);
        uint4 v4 = L4(p + d, g, r);
        uint v1 = select(ok, L1(select(ok, p + d + 4u, 0u), g, r), 0u);
        uint4 x4 = select(q, LX(select(q, (p + d) + uint4(0u, 4u, 8u, 12u), 0u), g, r), 0u);
        xout.Store4(o + 48u * k, v4);
        xout.Store4(o + 48u * k + 16u, x4);
        xout.Store4(o + 48u * k + 32u, uint4(v1, L2(p + d + 8u, g, r), d));
    }
}
"#;

const OLD_CALLS: &str = "uint L1(uint a, uint64_t g, uint4 r) { return xcl_mLoad(a); }\nuint2 L2(uint a, uint64_t g, uint4 r) { return xcl_mLoad2(a); }\nuint4 L4(uint a, uint64_t g, uint4 r) { return xcl_mLoad4(a); }\nuint4 LX(uint4 a, uint64_t g, uint4 r) { return xcl_loadx4(a); }\n";
const NEW_CALLS: &str = "uint L1(uint a, uint64_t g, uint4 r) { return xcl_mLoadG(a, g, r); }\nuint2 L2(uint a, uint64_t g, uint4 r) { return xcl_mLoad2G(a, g, r); }\nuint4 L4(uint a, uint64_t g, uint4 r) { return xcl_mLoad4G(a, g, r); }\nuint4 LX(uint4 a, uint64_t g, uint4 r) { return xcl_loadx4G(a, g, r); }\n";

/// The function of `src` whose text starts with `head`, to its closing brace.
fn function(src: &str, head: &str) -> String {
    let at = src.find(head).unwrap_or_else(|| panic!("{head}"));
    let mut depth = 0;
    for (k, c) in src[at..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' if depth == 1 => return format!("{}\n", &src[at..at + k + 1]),
            '}' => depth -= 1,
            _ => {}
        }
    }
    panic!("{head}: unclosed");
}

fn accessors() -> String {
    let all = super::emit::plain_mem_accessors(4, true);
    let heads = [
        "uint4 xcl_group_pages(", "uint2 xcl_group_at(", "uint xcl_mLoad(", "uint2 xcl_mLoad2(", "uint4 xcl_mLoad4(", "uint4 xcl_loadx4(",
        "uint xcl_mLoadG(", "uint2 xcl_mLoad2G(", "uint4 xcl_mLoad4G(", "uint4 xcl_loadx4G(",
    ];
    heads.iter().map(|h| function(&all, h)).collect()
}

fn program(accessors: &str, calls: &str) -> interp::Program {
    let src = format!(
        "static uint xcl_mem_views = 0u, xcl_uma_base = 0u, xcl_windows = 0u, xcl_w = 0u;\n{}\n{}\n{SETUP}\n{accessors}\n{calls}\n{DRIVER}",
        super::GPU_PAGE_DIRECTORY_HLSL,
        super::emit::PHYSICAL_ACCESS
    );
    interp::load(&xcl_dxil::compile(&src, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{src}"))).unwrap()
}

fn verdict(accessors: &str) -> Verdict {
    interp::diff(&program(accessors, OLD_CALLS), &program(accessors, NEW_CALLS), 0..24, &Config::default()).0
}

#[test]
fn grouped_loads_read_what_each_load_alone_reads() {
    let acc = accessors();
    assert!(acc.contains("xcl_group_at(v3, g, r)") && !acc.contains(" ? "));
    assert_eq!(verdict(&acc), Verdict::Equal);
}

#[test]
fn a_wrong_group_page_offset_or_miss_is_caught() {
    let acc = accessors();
    for (from, to) in [
        ("s + 0x10001ull", "s + 0x10000ull"),
        ("y - 1u + ((uint)va & 65535u)", "y + ((uint)va & 65535u)"),
        ("select(k == 1ull, r.z,", "select(k == 2ull, r.z,"),
        ("select(y == 0u, 0u, y - 1u", "select(false, 0u, y - 1u"),
        ("select(a.y == 0u, z.x, q1.x)", "select(a.y != 0u, z.x, q1.x)"),
        ("uint2 z = xcl_resolve_va(xcl_operand_va(0u))", "uint2 z = xcl_resolve_va(xcl_operand_va(4u))"),
        ("uint64_t k = (va - (g & ~0xffffull)) >> 16", "uint64_t k = (va >> 16) - (g >> 16)"),
        ("uint2 p2 = xcl_group_at(xcl_operand_va(a2), g, r)", "uint2 p2 = xcl_group_at(xcl_operand_va(a1), g, r)"),
    ] {
        assert!(acc.contains(from), "{from}");
        assert!(matches!(verdict(&acc.replacen(from, to, 1)), Verdict::Different(_)), "mutant {to} not caught");
    }
}

type Inst = (&'static str, &'static str, Vec<(&'static str, u64)>);

fn code(insts: &[Inst]) -> Vec<u8> {
    insts.iter().flat_map(|(n, e, f)| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

/// Constant-offset loads of every grouped kind from one base, a base rewritten between two
/// loads, and an offset past a group's 64 KiB; every loaded register stored.
fn grouped_program() -> Vec<u8> {
    let wait = || ("S_WAITCNT", "ENC_SOPP", vec![("SIMM16", 0)]);
    let buf = |n: &'static str, d: u64, o: u64| -> Inst { (n, "ENC_MUBUF", vec![("VDATA", d), ("VADDR", 40), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", o), ("OFFEN", 1)]) };
    let mut p: Vec<Inst> = vec![
        ("S_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 8), ("SBASE", 0), ("OFFSET", 0)]),
        ("S_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 12), ("SBASE", 0), ("OFFSET", 0x40)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", vec![("SDATA", 16), ("SBASE", 0), ("OFFSET", 0x10010)]),
        wait(),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 40), ("SRC0", 132), ("VSRC1", 0)]),
        buf("BUFFER_LOAD_DWORD", 1, 4),
        buf("BUFFER_LOAD_DWORDX4", 2, 16),
        ("S_BUFFER_LOAD_DWORDX4", "ENC_SMEM", vec![("SDATA", 20), ("SBASE", 4), ("OFFSET", 0x20)]),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", vec![("SDATA", 24), ("SBASE", 4), ("OFFSET", 0x30)]),
        wait(),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 40), ("SRC0", 0xFF), ("VSRC1", 0)]),
        buf("BUFFER_LOAD_DWORDX4", 6, 32),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 42), ("SRC0", 130), ("VSRC1", 0)]),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 43), ("SRC0", 9)]),
        ("GLOBAL_LOAD_DWORDX4", "ENC_FLAT_GLBL", vec![("VDST", 10), ("ADDR", 42), ("SADDR", 0x7D), ("OFFSET", 0)]),
        ("GLOBAL_LOAD_DWORD", "ENC_FLAT_GLBL", vec![("VDST", 14), ("ADDR", 42), ("SADDR", 0x7D), ("OFFSET", 32)]),
        wait(),
    ];
    for (k, s) in [12u64, 13, 16, 17, 20, 23, 24].into_iter().enumerate() {
        p.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 15 + k as u64), ("SRC0", s)]));
    }
    p.push(("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 44), ("SRC0", 134), ("VSRC1", 0)]));
    for r in 1..22u64 {
        p.push(("BUFFER_STORE_DWORD", "ENC_MUBUF", vec![("VDATA", r), ("VADDR", 44), ("SRSRC", 2), ("SOFFSET", 128), ("OFFSET", 4 * r), ("OFFEN", 1)]));
    }
    p.push(("S_ENDPGM", "ENC_SOPP", vec![]));
    let mut c = code(&p);
    // the literal of the second v_lshlrev (SRC0 0xFF): a voffset 64 KiB away
    let at = c.len() - code(&p[p.iter().position(|i| i.2.contains(&("SRC0", 0xFF))).unwrap()..]).len() + 4;
    c.splice(at..at, 17u32.to_le_bytes());
    c
}

fn translate_cs(code: &[u8], groups: bool) -> String {
    let cs = super::CsInterface { threads: [64, 1, 1], user_sgprs: 4, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    super::emit::NO_PAGE_GROUPS.set(!groups);
    let t = translate(code, &Options { mode: Mode::Compute(cs), wave64: false, split_wave64: true, mem_windows: 4, ..Default::default() });
    super::emit::NO_PAGE_GROUPS.set(false);
    let t = t.unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl.replace("vector<uint64_t, 4>(", "uint64_t4(")
}

fn load(hlsl: &str) -> interp::Program {
    interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap()
}

#[test]
fn translated_loads_with_page_groups_compute_the_same() {
    let c = grouped_program();
    let (old, new) = (translate_cs(&c, false), translate_cs(&c, true));
    assert!(!old.contains("xcl_group_pages(") && !old.contains("G("));
    // the s_load pair, buffer loads at 4 and 16, the s_buffer pair, the x4 after the voffset
    // moved (its own group), the global pair; the s_load past 64 KiB stays alone
    let decls = new.lines().filter(|l| l.contains("= xcl_group_pages(t")).count();
    assert_eq!(decls, 5, "{new}");
    assert!(new.contains("xcl_loadx4G(") && new.contains("xcl_mLoad4G(") && new.contains("xcl_mLoadG(") && new.contains("xcl_vLoadG("));
    assert_eq!(interp::diff(&load(&old), &load(&new), 0..24, &Config::default()).0, Verdict::Equal);
}
