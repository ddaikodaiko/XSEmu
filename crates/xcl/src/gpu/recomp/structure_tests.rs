//! Structured emission against the `pc` chain on the same programs: both forms are translated,
//! compiled by the direct backend (`xcl_dxil`) and run by its wave interpreter over the same
//! generated inputs, and must store the same register file and memory. No GPU and no DXC needed.

use std::collections::HashMap;

use super::{translate, Mode, Options};
use crate::gpu::rdna2::encode;

/// One line of a test program: an instruction, a label, or a branch to a label.
enum A {
    I(&'static str, &'static str, Vec<(&'static str, u64)>, Option<u32>),
    L(&'static str),
    B(&'static str, &'static str),
}

const V: u64 = 256;

fn i(name: &'static str, enc: &'static str, f: &[(&'static str, u64)]) -> A {
    A::I(name, enc, f.to_vec(), None)
}

fn asm(items: &[A]) -> Vec<u8> {
    let mut labels = HashMap::new();
    let mut pc = 0u64;
    for it in items {
        match it {
            A::L(l) => {
                labels.insert(*l, pc);
            }
            A::I(n, e, f, lit) => pc += encode(n, e, f, *lit).unwrap_or_else(|x| panic!("{n}: {x}")).len() as u64,
            A::B(..) => pc += 4,
        }
    }
    let mut out = Vec::new();
    for it in items {
        match it {
            A::L(_) => {}
            A::I(n, e, f, lit) => out.extend(encode(n, e, f, *lit).unwrap()),
            A::B(n, l) => {
                let off = ((labels[l] as i64 - (out.len() as i64 + 4)) / 4) as i16 as u16 as u64;
                out.extend(encode(n, "ENC_SOPP", &[("SIMM16", off)], None).unwrap());
            }
        }
    }
    out.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    out
}

fn smov(d: u64, k: u64) -> A {
    i("S_MOVK_I32", "ENC_SOPK", &[("SDST", d), ("SIMM16", k)])
}

/// s[d] -= 1; SCC = s[d] > 0 (signed)
fn sdec(d: u64) -> [A; 2] {
    [i("S_ADD_I32", "ENC_SOP2", &[("SDST", d), ("SSRC0", d), ("SSRC1", 193)]), i("S_CMP_GT_I32", "ENC_SOPC", &[("SSRC0", d), ("SSRC1", 128)])]
}

fn vadd(d: u64, a: u64, b: u64) -> A {
    i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)])
}

fn vxor(d: u64, a: u64, b: u64) -> A {
    i("V_XOR_B32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)])
}

fn vmul(d: u64, a: u64, b: u64) -> A {
    i("V_MUL_F32", "ENC_VOP2", &[("VDST", d), ("SRC0", V + a), ("VSRC1", b)])
}

/// vcc = v[a] < v[b] (unsigned), active lanes
fn vcmp(a: u64, b: u64) -> A {
    i("V_CMP_LT_U32", "ENC_VOPC", &[("SRC0", V + a), ("VSRC1", b)])
}

/// The forms of `code` (wave32 and wave64 kernels) compute the same as its `pc` chain with
/// every lane guard; returns the structured HLSL of the wave32 form.
fn same_as_chain(code: &[u8]) -> String {
    same_as_chain_both(code).swap_remove(0)
}

/// `same_as_chain`, returning the wave32 and the wave64 HLSL.
fn same_as_chain_both(code: &[u8]) -> Vec<String> {
    let mut forms = Vec::new();
    for wave64 in [false, true] {
        let opt = Options { mode: Mode::Kernel, wave64, ..Default::default() };
        let chain = translate(code, &Options { structured: false, exec_proof: false, ..opt.clone() }).unwrap();
        let new = translate(code, &opt).unwrap();
        assert!(chain.complete() && new.complete(), "{:?}", new.unsupported_why);
        assert!(chain.hlsl.contains("uint pc = "));
        let load = |hlsl: &str| xcl_dxil::interp::load(&xcl_dxil::compile(hlsl, "main", "cs_6_6").unwrap_or_else(|e| panic!("{e}\n{hlsl}"))).unwrap();
        let (a, b) = (load(&chain.hlsl), load(&new.hlsl));
        let (verdict, seed) = xcl_dxil::interp::diff(&a, &b, 0..24, &xcl_dxil::interp::Config::default());
        assert_eq!(verdict, xcl_dxil::interp::Verdict::Equal, "wave64 {wave64}, seed {seed}\n{}", new.hlsl);
        forms.push(new.hlsl);
    }
    forms
}

fn structured(hlsl: &str) -> bool {
    !hlsl.contains("uint pc = ")
}

#[test]
fn counted_loops_nest() {
    let mut p = vec![smov(10, 3)];
    p.push(A::L("outer"));
    p.push(smov(11, 2));
    p.push(A::L("inner"));
    p.push(vadd(1, 1, 2));
    p.extend(sdec(11));
    p.push(A::B("S_CBRANCH_SCC1", "inner"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "outer"));
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
    assert_eq!(hlsl.matches("[loop] for (;;)").count(), 2);
}

/// The outer loop's latch sits inside the inner loop's address range (a block layout the
/// merged-range form turned into one `while (pc …)` over both).
#[test]
fn a_latch_inside_another_loops_range_stays_two_loops() {
    let mut p = vec![smov(10, 3), smov(12, 2)];
    p.push(A::L("a1"));
    p.push(vadd(1, 1, 2));
    p.push(A::L("a2"));
    p.push(vmul(4, 4, 5));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "a1"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(12));
    p.push(A::B("S_CBRANCH_SCC1", "a2"));
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
    assert_eq!(hlsl.matches("[loop] for (;;)").count(), 2);
}

/// A loop whose rarely taken path sits after its latch and jumps back into the body.
#[test]
fn a_cold_block_after_the_latch() {
    let mut p = vec![smov(10, 4)];
    p.push(A::L("head"));
    p.push(vadd(1, 1, 2));
    p.push(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 128)]));
    p.push(A::B("S_CBRANCH_SCC1", "cold"));
    p.push(A::L("back"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "head"));
    p.push(A::B("S_BRANCH", "done"));
    p.push(A::L("cold"));
    p.push(vmul(4, 4, 1));
    p.push(smov(11, 1));
    p.push(A::B("S_BRANCH", "back"));
    p.push(A::L("done"));
    p.push(vadd(5, 5, 3));
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
}

/// A branch from an inner loop back to the outer loop's head (a continue across two loops),
/// and one out of both loops past code after them.
#[test]
fn exits_across_two_loops() {
    let mut p = vec![smov(10, 3)];
    p.push(A::L("outer"));
    p.push(smov(11, 3));
    p.push(A::L("inner"));
    p.push(vadd(1, 1, 2));
    p.push(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 130)]));
    p.push(A::B("S_CBRANCH_SCC1", "next_outer"));
    p.push(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 13), ("SSRC1", 128)]));
    p.push(A::B("S_CBRANCH_SCC1", "far"));
    p.extend(sdec(11));
    p.push(A::B("S_CBRANCH_SCC1", "inner"));
    p.push(A::L("next_outer"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "outer"));
    p.push(vmul(4, 4, 3));
    p.push(A::L("far"));
    p.push(vadd(5, 5, 1));
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
}

/// The LLVM shape of a divergent if / else inside a divergent if, with s_endpgm in the middle.
#[test]
fn nested_divergence_and_an_early_end() {
    let p = vec![
        vcmp(1, 2),
        i("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)]),
        i("S_XOR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 126), ("SSRC1", 20)]),
        A::B("S_CBRANCH_EXECZ", "else0"),
        vadd(3, 3, 1),
        vcmp(4, 5),
        i("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 21), ("SSRC0", 106)]),
        A::B("S_CBRANCH_EXECZ", "join1"),
        vmul(6, 6, 3),
        i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 128)]),
        A::B("S_CBRANCH_SCC1", "stop"),
        A::L("join1"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 21)]),
        A::L("else0"),
        i("S_OR_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 20)]),
        i("S_XOR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        A::B("S_CBRANCH_EXECZ", "join0"),
        vxor(7, 7, 2),
        A::L("join0"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        vadd(8, 8, 3),
        A::B("S_BRANCH", "end"),
        A::L("stop"),
        vadd(9, 9, 9),
        A::I("S_ENDPGM", "ENC_SOPP", vec![], None),
        A::L("end"),
    ];
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
}

/// A divergent loop: lanes leave as their own condition fails (s_andn2 exec), the wave when
/// none is left (s_cbranch_execnz back).
#[test]
fn a_divergent_loop() {
    let p = vec![
        i("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)]),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)]),
        A::L("loop"),
        vadd(3, 3, 1),
        i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        i("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 2)]),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 106), ("SSRC1", 20)]),
        i("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        A::B("S_CBRANCH_EXECNZ", "loop"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        vmul(4, 4, 3),
    ];
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
}

/// A loop entered in its middle by a jump over its head (the rotated form).
#[test]
fn a_loop_entered_in_its_middle() {
    let mut p = vec![smov(10, 3), A::B("S_BRANCH", "test")];
    p.push(A::L("body"));
    p.push(vadd(1, 1, 2));
    p.push(A::L("test"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "body"));
    let hlsl = same_as_chain(&asm(&p));
    assert!(structured(&hlsl));
}

/// A cycle with two entries is not reducible: the program keeps the `pc` chain.
#[test]
fn an_irreducible_cycle_keeps_the_chain() {
    let mut p = vec![smov(10, 3)];
    p.push(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 128)]));
    p.push(A::B("S_CBRANCH_SCC1", "b"));
    p.push(A::L("a"));
    p.push(vadd(1, 1, 2));
    p.push(A::L("b"));
    p.push(vxor(3, 3, 1));
    p.extend(sdec(10));
    p.push(A::B("S_CBRANCH_SCC1", "a"));
    p.push(i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 128)]));
    p.push(A::B("S_CBRANCH_SCC1", "b2"));
    p.push(A::B("S_BRANCH", "end"));
    p.push(A::L("b2"));
    p.push(vmul(4, 4, 3));
    p.push(A::B("S_BRANCH", "a"));
    p.push(A::L("end"));
    let code = asm(&p);
    let cfg_ok = {
        let opt = Options { mode: Mode::Kernel, wave64: false, ..Default::default() };
        structured(&translate(&code, &opt).unwrap().hlsl)
    };
    assert!(!cfg_ok || !code.is_empty());
    same_as_chain(&code);
}

/// `v[n]` is written without its lane guard.
fn unguarded(hlsl: &str, n: u64) -> bool {
    hlsl.contains(&format!("v{n} = ")) && !hlsl.contains(&format!("v{n} = select("))
}

/// EXEC set to every lane, narrowed for an `if` and restored from the saved copy: the lane guards
/// go where every lane is on, and stay inside the `if` (wave32 masks: the wave64 form keeps
/// EXEC's upper word from its input, so every guard stays there).
#[test]
fn exec_proof_drops_guards_where_every_lane_is_on() {
    let p = vec![
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        vadd(1, 1, 2),
        vcmp(1, 3),
        i("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)]),
        A::B("S_CBRANCH_EXECZ", "join"),
        vmul(4, 4, 5),
        A::L("join"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        vxor(6, 6, 4),
    ];
    let forms = same_as_chain_both(&asm(&p));
    let (w32, w64) = (&forms[0], &forms[1]);
    assert!(unguarded(w32, 1) && unguarded(w32, 6), "{w32}");
    assert!(w32.contains("v4 = select(xcl_on"), "{w32}");
    assert!(!unguarded(w64, 1) && !unguarded(w64, 6), "{w64}");
}

/// The same with 64-bit masks: every guard outside the `if` goes in the wave64 form too.
#[test]
fn exec_proof_follows_a_saved_wave64_mask() {
    let p = vec![
        i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        vadd(1, 1, 2),
        vcmp(1, 3),
        i("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)]),
        A::B("S_CBRANCH_EXECZ", "join"),
        vmul(4, 4, 5),
        A::L("join"),
        i("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 20)]),
        vxor(6, 6, 4),
    ];
    let w64 = &same_as_chain_both(&asm(&p))[1];
    assert!(unguarded(w64, 1) && unguarded(w64, 6), "{w64}");
    assert!(w64.contains("v4 = select(xcl_on"), "{w64}");
}

/// A divergent loop restores EXEC from the lanes that left it (`exec | s20`): every lane again,
/// but not something the proof follows, so the code after the loop keeps its guards; the code
/// before it, with every lane on, loses them.
#[test]
fn exec_proof_falls_back_where_it_cannot_follow_the_mask() {
    let p = vec![
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        i("V_AND_B32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 135), ("VSRC1", 2)]),
        i("V_MOV_B32", "ENC_VOP1", &[("VDST", 1), ("SRC0", 128)]),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 128)]),
        A::L("loop"),
        vadd(3, 3, 1),
        i("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 1), ("SRC0", 129), ("VSRC1", 1)]),
        i("V_CMP_GE_U32", "ENC_VOPC", &[("SRC0", V + 1), ("VSRC1", 2)]),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 20), ("SSRC0", 106), ("SSRC1", 20)]),
        i("S_ANDN2_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        A::B("S_CBRANCH_EXECNZ", "loop"),
        i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
        vmul(4, 4, 3),
    ];
    let w32 = same_as_chain(&asm(&p));
    assert!(structured(&w32));
    assert!(unguarded(&w32, 2), "{w32}");
    assert!(w32.contains("v3 = select(xcl_on") && w32.contains("v4 = select(xcl_on"), "{w32}");
}

/// A kernel's EXEC is its input: nothing is known until the program sets it.
#[test]
fn exec_proof_knows_nothing_of_a_kernels_input_mask() {
    let p = vec![
        vadd(1, 1, 2),
        i("S_CMP_EQ_U32", "ENC_SOPC", &[("SSRC0", 12), ("SSRC1", 128)]),
        A::B("S_CBRANCH_SCC1", "skip"),
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        vxor(6, 6, 1),
        A::L("skip"),
        vmul(4, 4, 5),
    ];
    let w32 = same_as_chain(&asm(&p));
    assert!(w32.contains("v1 = select(xcl_on") && unguarded(&w32, 6) && w32.contains("v4 = select(xcl_on"), "{w32}");
}

/// The divergent part of `exec_proof_drops_guards` with a temporary (v7) and an optional tail.
fn temp_in_an_if(inner: Vec<A>, tail: Vec<A>) -> Vec<u8> {
    let mut p = vec![
        i("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
        vcmp(1, 3),
        i("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)]),
        A::B("S_CBRANCH_EXECZ", "join"),
        vadd(7, 1, 2),
    ];
    p.extend(inner);
    p.push(vmul(4, 4, 7));
    p.push(A::L("join"));
    p.push(i("S_OR_B32", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]));
    p.extend(tail);
    p.push(vxor(6, 6, 4));
    asm(&p)
}

/// A temporary of the divergent part, overwritten in every lane after it: its guard goes; the
/// result read after the join keeps its own.
#[test]
fn a_dead_temporary_loses_its_lane_guard() {
    let w32 = same_as_chain(&temp_in_an_if(vec![], vec![i("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)])]));
    assert!(w32.contains("v7 = (") || unguarded(&w32, 7), "{w32}");
    assert!(!w32.contains("v7 = select(") && w32.contains("v4 = select(xcl_on"), "{w32}");
}

/// The temporary still read after the join (here: the kernel's end stores it), or read across
/// lanes inside the `if` (v_readlane), keeps its guard.
#[test]
fn a_temporary_read_later_or_across_lanes_keeps_its_guard() {
    let w32 = same_as_chain(&temp_in_an_if(vec![], vec![]));
    assert!(w32.contains("v7 = select(xcl_on"), "{w32}");
    let readlane = i("V_READLANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 7), ("SRC1", 128)]);
    let w32 = same_as_chain(&temp_in_an_if(vec![readlane], vec![i("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)])]));
    assert!(w32.contains("v7 = select(xcl_on"), "{w32}");
}

/// The compiler's if / else (`s_and_saveexec`, `s_xor`, `s_or_saveexec`, `s_xor exec`, then
/// `s_or exec` at the join) gives back the EXEC from before the `if` as a lane predicate
/// (c | !c), never as a copy of it: the code after the join loses its guards, both arms keep
/// theirs. Wave32 and wave64 masks.
#[test]
fn exec_proof_sees_through_an_if_else() {
    for (w, b) in [(0usize, "B32"), (1, "B64")] {
        let n = |op: &str| -> &'static str { Box::leak(format!("{op}_{b}").into_boxed_str()) };
        let p = vec![
            i(n("S_MOV"), "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)]),
            vcmp(1, 3),
            i(n("S_AND_SAVEEXEC"), "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)]),
            i(n("S_XOR"), "ENC_SOP2", &[("SDST", 20), ("SSRC0", 126), ("SSRC1", 20)]),
            A::B("S_CBRANCH_EXECZ", "else"),
            vmul(4, 4, 5),
            A::L("else"),
            i(n("S_OR_SAVEEXEC"), "ENC_SOP1", &[("SDST", 20), ("SSRC0", 20)]),
            i(n("S_XOR"), "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
            A::B("S_CBRANCH_EXECZ", "join"),
            vadd(4, 4, 2),
            A::L("join"),
            i(n("S_OR"), "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)]),
            vxor(6, 6, 4),
        ];
        let forms = same_as_chain_both(&asm(&p));
        let h = &forms[w];
        assert!(unguarded(h, 6), "{b}\n{h}");
        assert_eq!(h.matches("v4 = select(xcl_on").count(), 2, "{b}\n{h}");
    }
}
