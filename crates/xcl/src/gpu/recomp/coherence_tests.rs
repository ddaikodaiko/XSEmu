use super::{translate, CsInterface, Mode, Options};
use crate::gpu::rdna2::encode;

type Inst = (&'static str, &'static str, &'static [(&'static str, u64)]);

const PLAIN: Inst = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 8), ("SOFFSET", 125)]);
const FLAGGED: Inst = ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 9), ("SOFFSET", 125), ("OFFSET", 4), ("GLC", 1)]);
const ATOMIC: Inst = ("GLOBAL_ATOMIC_ADD", "ENC_FLAT_GLBL", &[("VDST", 5), ("ADDR", 0), ("DATA", 2), ("SADDR", 2), ("GLC", 1)]);

fn program(insts: &[Inst]) -> Vec<u8> {
    insts.iter().copied().chain([("S_ENDPGM", "ENC_SOPP", &[][..])])
        .flat_map(|(name, enc, fields)| encode(name, enc, fields, None).unwrap_or_else(|e| panic!("{name}: {e}")))
        .collect()
}

fn options(mem_windows: u32, wave64: bool, split_wave64: bool, legacy_compute_wave64: bool) -> Options {
    let threads = if wave64 { 64 } else { 32 };
    let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 4, tgid: [false; 3], tg_size: false, tid_components: 1 };
    Options {
        mode: Mode::Compute(cs), mem_windows, wave64, split_wave64, legacy_compute_wave64,
        compute_per_access_coherence: true, guard_trip_counters: false, ..Default::default()
    }
}

fn hlsl(insts: &[Inst], opt: &Options) -> String {
    let t = translate(&program(insts), opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

fn legacy_hlsl(insts: &[Inst], opt: &Options) -> String {
    hlsl(insts, &Options { compute_per_access_coherence: false, ..opt.clone() })
}

#[test]
fn compute_per_access_coherence_caches_only_before_first_flagged_read() {
    let wide = ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 12), ("SOFFSET", 125), ("OFFSET", 16)][..]);
    let wait = ("S_WAITCNT", "ENC_SOPP", &[("SIMM16", 0)][..]);
    for windows in [1, 4] {
        for (wave64, split, legacy) in [(false, false, false), (true, false, false), (true, true, false), (true, true, true)] {
            let opt = options(windows, wave64, split, legacy);
            let h = hlsl(&[PLAIN, wide, wait, FLAGGED, wait, PLAIN, wide], &opt);
            let (prelude, body) = h.split_once("void main(").unwrap();
            let call = body.lines().filter_map(|line| {
                let (_, tail) = line.split_once(" = xcl_m")?;
                Some(tail.split_once('(')?.0)
            }).collect::<Vec<_>>();
            let grouped = windows > 1 && !(split && !legacy);
            let after = if grouped { ["LoadG", "Load4G"] } else { ["Load", "Load4"] };
            assert_eq!(call, ["_cachedLoad", "_cachedLoad4", "Load", after[0], after[1]]);
            let view = if windows == 1 { "xcl_mem_views" } else { "xcl_mem_views + xcl_w" };
            if grouped {
                assert!(prelude.contains(&format!("xcl_mLoadG(uint a, uint64_t g, uint4 r) {{ globallycoherent RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            }
            assert!(prelude.contains(&format!("xcl_m_cachedLoad(uint a) {{ RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            assert!(prelude.contains(&format!("xcl_mLoad(uint a) {{ globallycoherent RWByteAddressBuffer xcl_mb = ResourceDescriptorHeap[{view}]")));
            assert_eq!(hlsl(&[FLAGGED, PLAIN], &opt), legacy_hlsl(&[FLAGGED, PLAIN], &opt));
        }
    }
}

#[test]
fn compute_per_access_coherence_preserves_whole_shader_fallbacks() {
    let opt = options(4, false, false, false);
    let baseline = hlsl(&[PLAIN], &opt);
    assert!(!baseline.contains("globallycoherent"));
    assert_eq!(baseline, legacy_hlsl(&[PLAIN], &opt));
    let cases: &[Inst] = &[
        ("S_DCACHE_INV", "ENC_SMEM", &[]),
        ("S_GL1_INV", "ENC_SMEM", &[]),
        ("BUFFER_GL0_INV", "ENC_MUBUF", &[]),
        ("BUFFER_GL1_INV", "ENC_MUBUF", &[]),
        ("IMAGE_LOAD", "ENC_MIMG", &[("DMASK", 1), ("DIM", 1)]),
        ("FLAT_LOAD_DWORD", "ENC_FLAT", &[("VDST", 2)]),
        ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 125)]),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 4)]),
        ("S_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SOFFSET", 125), ("SBASE", 6)]),
        ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 0), ("SOFFSET", 125)]),
    ];
    for &inst in cases {
        let code = [PLAIN, FLAGGED, inst, PLAIN];
        let h = hlsl(&code, &opt);
        assert!(h.contains("globallycoherent"));
        assert!(!h.contains("_cachedLoad"), "{inst:?}");
        assert!(h == legacy_hlsl(&code, &opt), "{inst:?}");
    }
    let kernel = Options { mode: Mode::Kernel, ..opt };
    let h = hlsl(&[PLAIN, ATOMIC], &kernel);
    assert!(!h.contains("globallycoherent"));
    assert_eq!(h, legacy_hlsl(&[PLAIN, ATOMIC], &kernel));
}

#[test]
fn compute_per_access_coherence_keeps_writes_atomics_and_barriers() {
    let store = ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("DATA", 2), ("SADDR", 0)][..]);
    let byte_store = ("BUFFER_STORE_BYTE", "ENC_MUBUF", &[("VDATA", 2), ("SOFFSET", 128)][..]);
    let barrier = ("S_BARRIER", "ENC_SOPP", &[][..]);
    let cas = ("FLAT_ATOMIC_INC_X2", "ENC_FLAT", &[("VDST", 5), ("DATA", 2), ("GLC", 1)][..]);
    for windows in [1, 4] {
        let opt = options(windows, true, true, false);
        for inst in [ATOMIC, store, byte_store, barrier, cas] {
            let code = [PLAIN, inst, PLAIN, FLAGGED, PLAIN];
            let h = hlsl(&code, &opt);
            assert!(!h.contains("_cachedLoad"), "{}", inst.0);
            assert!(h == legacy_hlsl(&code, &opt), "{}", inst.0);
        }
    }
}

#[test]
fn compute_per_access_coherence_does_not_cache_branching_or_polling() {
    let opt = options(4, false, false, false);
    let branch = ("S_BRANCH", "ENC_SOPP", &[("SIMM16", 0)][..]);
    let cmp = ("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 9), ("SSRC1", 128)][..]);
    let back = ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xfffc)][..]);
    for code in [
        &[PLAIN, branch, FLAGGED, PLAIN][..],
        &[PLAIN, FLAGGED, cmp, back, PLAIN][..],
    ] {
        let h = hlsl(code, &opt);
        assert!(!h.contains("_cachedLoad"));
        assert!(h == legacy_hlsl(code, &opt));
    }
}
