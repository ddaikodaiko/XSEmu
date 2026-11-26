//! Pixel lane reads whose source lies in the reader's own quad (DPP16 quad_perm and row_xmask
//! below 4, DPP8 and ds_swizzle patterns that stay in the quad) as quad reads, against the
//! earlier form (`rla`: wave reads, then the quad reads for the reader's own quad), on the same
//! programs: both forms are compiled by the direct backend (`xcl_dxil`) and run by its
//! interpreter as quads of pixels, and must export the same.

use super::{translate, Mode, Options, PsInterface};
use crate::gpu::rdna2::encode;
use xcl_dxil::interp::{self, Config, Verdict};

fn el(name: &str, sv: u32, reg: u32) -> crate::gpu::dxbc::SigElement {
    crate::gpu::dxbc::SigElement { name: name.into(), index: 0, system_value: sv, component_type: 3, register: reg, mask: 0xF, rw_mask: 0xF, stream: 0 }
}

type Inst = (&'static str, &'static str, Vec<(&'static str, u64)>);

/// COLOR into v4..v7, `ops` (results in v8..v11), exported to SV_Target.
fn program(ops: &[Inst]) -> Vec<u8> {
    let mut v: Vec<Inst> = Vec::new();
    for c in 0..4u64 {
        v.push(("V_INTERP_P1_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 0), ("ATTR", 0), ("ATTRCHAN", c)]));
        v.push(("V_INTERP_P2_F32", "ENC_VINTRP", vec![("VDST", 4 + c), ("VSRC", 1), ("ATTR", 0), ("ATTRCHAN", c)]));
    }
    for d in 8..12u64 {
        v.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", d), ("SRC0", 256 + d - 4)]));
    }
    v.extend(ops.iter().cloned());
    v.push(("EXP", "ENC_EXP", vec![("TGT", 0), ("EN", 0xF), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11), ("DONE", 1), ("VM", 1)]));
    v.push(("S_ENDPGM", "ENC_SOPP", vec![]));
    v.iter().flat_map(|(n, e, f)| encode(n, e, f, None).unwrap_or_else(|x| panic!("{n}: {x}"))).collect()
}

fn dpp(dst: u64, src: u64, ctrl: u64, row: u64, bound: u64) -> Inst {
    ("V_MOV_B32", "VOP1_VOP_DPP16", vec![("VDST", dst), ("SRC0", 0xFA), ("VSRC0", src), ("DPP_CTRL", ctrl), ("ROW_MASK", row), ("BANK_MASK", 0xF), ("BOUND_CTRL", bound)])
}

fn add_dpp(dst: u64, src: u64, ctrl: u64) -> Inst {
    ("V_ADD_F32", "VOP2_VOP_DPP16", vec![("VDST", dst), ("SRC0", 0xFA), ("VSRC0", src), ("VSRC1", dst), ("DPP_CTRL", ctrl), ("ROW_MASK", 0xF), ("BANK_MASK", 0xF)])
}

fn dpp8(dst: u64, src: u64, sel: [u64; 8]) -> Inst {
    const F: [&str; 8] = ["LANE_SEL_0", "LANE_SEL_1", "LANE_SEL_2", "LANE_SEL_3", "LANE_SEL_4", "LANE_SEL_5", "LANE_SEL_6", "LANE_SEL_7"];
    let mut f = vec![("VDST", dst), ("SRC0", 0xE9), ("VSRC0", src)];
    f.extend(F.iter().zip(sel).map(|(&n, s)| (n, s)));
    ("V_MOV_B32", "VOP1_VOP_DPP8", f)
}

fn swizzle(dst: u64, src: u64, offset: u64) -> [Inst; 2] {
    [("DS_SWIZZLE_B32", "ENC_DS", vec![("VDST", dst), ("ADDR", src), ("OFFSET0", offset & 0xFF), ("OFFSET1", offset >> 8)]), ("S_WAITCNT", "ENC_SOPP", vec![("SIMM16", 0)])]
}

fn translate_ps(code: &[u8], wave64: bool, split: bool, quad: bool) -> String {
    let ps = PsInterface { inputs: vec![el("SV_Position", 1, 0), el("COLOR", 0, 1)], outputs: vec![el("SV_Target", 64, 0)], input_ena: 0x302, attrs: None };
    super::emit::NO_QUAD_READS.set(!quad);
    let t = translate(code, &Options { mode: Mode::Pixel(ps), wave64, split_wave64: split, ..Default::default() });
    super::emit::NO_QUAD_READS.set(false);
    let t = t.unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    t.hlsl
}

fn load(hlsl: &str) -> interp::Program {
    interp::load(&xcl_dxil::compile(hlsl, "main", "ps_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap()
}

fn verdict(a: &str, b: &str) -> Verdict {
    interp::diff(&load(a), &load(b), 0..24, &Config::default()).0
}

/// Each case: its instructions and what its quad-read form must contain.
fn cases() -> Vec<(Vec<Inst>, &'static str)> {
    vec![
        (vec![dpp(8, 4, 0xB1, 0xF, 0), dpp(9, 5, 0x4E, 0xF, 1), dpp(10, 6, 0x1B, 0xF, 0), dpp(11, 7, 0xAA, 0xF, 0)], "QuadReadLaneAt("),
        (vec![dpp(8, 4, 0x93, 0xF, 0), dpp(9, 5, 0x161, 0xF, 0), dpp(10, 6, 0x162, 0xA, 1), add_dpp(11, 7, 0x163)], "QuadReadLaneAt("),
        (vec![dpp(8, 4, 0xE4, 0x5, 0), add_dpp(9, 5, 0x27), dpp(10, 6, 0x111, 0xF, 1), dpp(11, 7, 0x00, 0xF, 0)], "QuadReadLaneAt("),
        (vec![dpp8(8, 4, [1, 0, 3, 2, 5, 4, 7, 6]), dpp8(9, 5, [3, 3, 0, 1, 7, 7, 4, 5]), dpp8(10, 6, [7, 6, 5, 4, 3, 2, 1, 0])], "QuadReadLaneAt("),
        (swizzle(8, 4, 0x80B1).into_iter().chain(swizzle(9, 5, 0x041F)).chain(swizzle(10, 6, 0x8027)).chain(swizzle(11, 7, 0x101F)).collect(), "QuadReadLaneAt("),
    ]
}

#[test]
fn quad_local_lane_reads_as_quad_reads_compute_the_same() {
    for (ops, expect) in cases() {
        let code = program(&ops);
        for (wave64, split) in [(false, false), (true, false), (true, true)] {
            let (old, new) = (translate_ps(&code, wave64, split, false), translate_ps(&code, wave64, split, true));
            assert!(old.matches("QuadReadLaneAt(").count() < new.matches("QuadReadLaneAt(").count() && new.contains(expect) && !new.contains("QuadReadAcross"), "wave64 {wave64} split {split}\n{new}");
            assert_eq!(verdict(&old, &new), Verdict::Equal, "wave64 {wave64} split {split}\n{new}");
        }
    }
}

#[test]
fn a_wrong_quad_read_is_caught() {
    let code = program(&cases()[0].0);
    let (old, new) = (translate_ps(&code, false, false, false), translate_ps(&code, false, false, true));
    let mutant = new.replacen("select((lane_id & 1u) != 0u", "select((lane_id & 1u) == 0u", 1);
    assert_ne!(mutant, new);
    assert!(matches!(verdict(&old, &mutant), Verdict::Different(_)));
}
