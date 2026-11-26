use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::rc::Rc;

use crate::gpu::rdna2::{self, Decoded, Variant};
use crate::gpu::recomp::{self, DecodedProgram, Mode, MsInterface, Options};

use super::{
    decode_program, dxc, emit_program, hex, mem_windows, stage_psv, Trace, TraceScope, ACTIVE,
};

const PROFILE: &str = "cs_6_6";
const BASE: usize = 12288;
const ROW_BYTES: usize = 64;
const MARK: &str = "/* xcl_producer:";

type Fields = &'static [(&'static str, u64)];
const EXPORT_FLAGS: Fields = &[("COMPR", 0), ("DONE", 1), ("VM", 0)];
const BUFFER_FLAGS: Fields = &[
    ("OFFEN", 1),
    ("IDXEN", 0),
    ("LDS", 0),
    ("TFE", 0),
    ("GLC", 0),
    ("DLC", 0),
    ("SLC", 0),
    ("SRSRC", 8),
    ("SOFFSET", 128),
];
const SITES: [(&str, Fields); 14] = [
    (
        "S_LOAD_DWORDX4",
        &[
            ("SDATA", 32),
            ("SBASE", 24),
            ("SOFFSET", 125),
            ("OFFSET", 160),
        ],
    ),
    (
        "BUFFER_LOAD_DWORDX2",
        &[("VDATA", 1), ("VADDR", 1), ("OFFSET", 12)],
    ),
    ("S_WAITCNT", &[("SIMM16", 0xc07f)]),
    ("V_READFIRSTLANE_B32", &[("VDST", 52), ("SRC0", 256 + 25)]),
    ("V_READFIRSTLANE_B32", &[("VDST", 53), ("SRC0", 256 + 3)]),
    (
        "BUFFER_LOAD_DWORD",
        &[("VDATA", 2), ("VADDR", 2), ("OFFSET", 0)],
    ),
    ("S_WAITCNT", &[("SIMM16", 0x3f70)]),
    ("S_CMP_EQ_U32", &[("SSRC0", 52), ("SSRC1", 128)]),
    ("S_SENDMSG", &[("SIMM16", 9)]),
    ("V_BFREV_B32", &[("VDST", 1), ("SRC0", 129)]),
    (
        "EXP",
        &[
            ("TGT", 12),
            ("EN", 0),
            ("VSRC0", 0),
            ("VSRC1", 0),
            ("VSRC2", 0),
            ("VSRC3", 0),
        ],
    ),
    (
        "EXP",
        &[
            ("TGT", 20),
            ("EN", 1),
            ("VSRC0", 1),
            ("VSRC1", 0),
            ("VSRC2", 0),
            ("VSRC3", 0),
        ],
    ),
    (
        "EXP",
        &[
            ("TGT", 20),
            ("EN", 1),
            ("VSRC0", 1),
            ("VSRC1", 0),
            ("VSRC2", 0),
            ("VSRC3", 0),
        ],
    ),
    (
        "EXP",
        &[
            ("TGT", 12),
            ("EN", 15),
            ("VSRC0", 67),
            ("VSRC1", 68),
            ("VSRC2", 69),
            ("VSRC3", 70),
        ],
    ),
];

pub(crate) fn sites() -> Result<[u64; 14], String> {
    let config = std::env::var("XCL_MESH_PRODUCER_SITES")
        .map_err(|_| "XCL_MESH_PRODUCER_SITES is required")?;
    let pcs: [u64; 14] = config
        .split(',')
        .map(hex)
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "expected 14 mesh producer PCs")?;
    if pcs.iter().any(|&pc| pc % 4 != 0 || pc > u32::MAX as u64)
        || !pcs.windows(2).all(|p| p[0] < p[1])
    {
        return Err("mesh producer PCs must be aligned and strictly increasing".into());
    }
    Ok(pcs)
}

pub(crate) struct Probe([u64; 14]);

impl Probe {
    fn verify(&self, program: &DecodedProgram<'_>) -> Result<(), String> {
        let find = |pc| program.insts.iter().find(|d| d.pc == pc);
        for (i, (&pc, (name, fields))) in self.0.iter().zip(SITES).enumerate() {
            let d = find(pc).ok_or_else(|| {
                format!("mesh producer PC {pc:#x} is not an instruction boundary")
            })?;
            let common = match name {
                "EXP" => EXPORT_FLAGS,
                "BUFFER_LOAD_DWORD" | "BUFFER_LOAD_DWORDX2" => BUFFER_FLAGS,
                _ => &[],
            };
            if d.inst.name != name
                || !matches!(d.variant, Variant::Plain)
                || !fields
                    .iter()
                    .chain(common)
                    .all(|&(field, want)| rdna2::field_value(d.enc, &d.words, field) == Some(want))
            {
                return Err(format!(
                    "mesh producer site {i} at {pc:#x} does not match {name} operands"
                ));
            }
        }
        for (before, after) in [(1, 2), (3, 4), (5, 6), (9, 10), (10, 11)] {
            let d = find(self.0[before]).unwrap();
            let next = self.0[after];
            if d.pc + d.len as u64 != next
                || program
                    .insts
                    .iter()
                    .any(|d| rdna2::branch_target(d) == Some(next))
            {
                return Err(format!(
                    "mesh producer site at {next:#x} is not the direct fallthrough"
                ));
            }
        }
        for (dest, src) in [(48, 17), (49, 128)] {
            if !program
                .insts
                .iter()
                .take_while(|d| d.pc < self.0[0])
                .any(|d| {
                    d.inst.name == "S_MOV_B32"
                        && rdna2::field_value(d.enc, &d.words, "SDST") == Some(dest)
                        && rdna2::field_value(d.enc, &d.words, "SSRC0") == Some(src)
                })
            {
                return Err("mesh producer is missing the entry pointer setup".into());
            }
        }
        Ok(())
    }

    pub(super) fn before(&self, d: &Decoded, packed: bool) -> Option<String> {
        if packed {
            return None;
        }
        let i = self.0.binary_search(&d.pc).ok()?;
        Some(format!("{MARK}{i} */"))
    }

    fn instrument(&self, baseline: &str, marked: &str, windows: u32) -> Result<String, String> {
        let lines: Vec<_> = marked.split_inclusive('\n').collect();
        let mut unmarked = String::with_capacity(baseline.len());
        let mut anchors = Vec::new();
        for (line, &text) in lines.iter().enumerate() {
            if let Some(marker) = text.trim().strip_prefix(MARK) {
                let index = marker
                    .strip_suffix(" */")
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&i| i < SITES.len())
                    .ok_or("invalid mesh producer HLSL marker")?;
                anchors.push((line, index));
            } else {
                unmarked.push_str(text);
            }
        }
        if unmarked != baseline
            || anchors
                .iter()
                .map(|&(_, i)| i)
                .collect::<BTreeSet<_>>()
                .len()
                != SITES.len()
        {
            return Err(
                "mesh producer markers did not cover the unchanged ordinary lowering".into(),
            );
        }

        let mut insert = BTreeMap::<usize, String>::new();
        let mut add = |line, text: String| insert.entry(line).or_default().push_str(&text);
        let find_one = |needle: &str| {
            let mut found = lines
                .iter()
                .enumerate()
                .filter(|&(_, &line)| line.starts_with(needle));
            let (index, _) = found.next()?;
            found.next().is_none().then_some(index)
        };
        let entry = find_one("    bool xcl_on = ")
            .ok_or("mesh producer cannot locate the ordinary Mesh body")?;
        let resource = find_one("RWByteAddressBuffer res : register(u1);\n")
            .ok_or("mesh producer cannot locate the ordinary record UAV declaration")?;
        let window = if windows > 1 { "xcl_w" } else { "0u" };
        let uma = if windows > 1 { "xcl_uma_base" } else { "0u" };
        add(
            resource + 1,
            "RWByteAddressBuffer probe : register(u2);\n".into(),
        );
        add(entry + 1, "    bool xcl_prod_group0 = all(gid == uint3(0u, 0u, 0u));\n    bool xcl_prod_lane0 = xcl_prod_group0 && gi == 0u && xcl_slot == 0u;\n    uint xcl_prod_seen = 0u;\n".into());
        add(
            entry + 1,
            row(
                0,
                0,
                &[
                    "xcl_slot",
                    "xcl_wave",
                    "gi",
                    "WaveGetLaneCount()",
                    "s17",
                    "xcl_mem_views",
                    uma,
                ],
            ),
        );

        for &(line, i) in &anchors {
            let pc = self.0[i];
            let at = |slot, data: &[&str]| row(slot, pc, data);
            if let Some((offset, active)) = match i {
                8 => Some((13248, "lane_id == 0u")),
                10 => Some((13252, "xcl_on")),
                11 => Some((13256, "xcl_on")),
                12 => Some((13260, "xcl_on")),
                13 => Some((13264, "xcl_on")),
                _ => None,
            } {
                add(line, format!("    if (xcl_prod_group0 && {active}) {{ uint xcl_prod_old; probe.InterlockedAdd({offset}u, 1u, xcl_prod_old); }}\n"));
            }
            let indent = lines[line].len() - lines[line].trim_start().len();
            let lowered = || {
                lines
                    .iter()
                    .enumerate()
                    .skip(line + 1)
                    .take(18)
                    .take_while(|&(_, &text)| {
                        text.len() - text.trim_start().len() == indent
                            && !text.trim_start().starts_with(MARK)
                    })
                    .map(|(n, text)| (n, text.trim()))
            };
            match i {
                0 => {
                    let addr = lowered()
                        .find_map(|(_, text)| {
                            temp(text, "uint ", "_a = (s48 + 0x000000A0u + 0u) & ~3u;")
                        })
                        .ok_or("mesh producer cannot locate the SMEM address temporary")?;
                    let last = format!("s35 = {addr}_g0.w;");
                    let (end, _) = lowered()
                        .find(|&(_, text)| text == last)
                        .ok_or("mesh producer cannot locate the loaded SRD assignment")?;
                    add(
                        end + 1,
                        at(
                            1,
                            &[
                                "s48",
                                "s49",
                                &format!("{addr}_a"),
                                &format!("s49 + (uint)({addr}_a < s48)"),
                                window,
                                "s32",
                                "s33",
                                "s34",
                                "s35",
                            ],
                        ),
                    );
                }
                1 | 5 => {
                    let meshlet = i == 1;
                    let addr = lowered()
                        .find_map(|(_, text)| {
                            temp(text, "uint ", "_stride = (s33 >> 16) & 0x3FFFu;")
                        })
                        .ok_or("mesh producer cannot locate the MUBUF address temporaries")?;
                    let load = if meshlet {
                        format!(
                            "uint4 {addr}_v = xcl_loadx4(select(and({addr}_act, {addr}_ok), ({addr}_a & ~3u) + uint4(0u, 4u, 4u, 4u), 0u));"
                        )
                    } else {
                        format!(
                            "uint {addr}_v = xcl_mLoad(select(and({addr}_act, {addr}_ok), {addr}_a & ~3u, 0u));"
                        )
                    };
                    let (end, _) = lowered()
                        .find(|&(_, text)| text == load)
                        .ok_or("mesh producer cannot locate the actual MUBUF load expression")?;
                    let voff = if meshlet { "v1" } else { "v2" };
                    let bounds = if meshlet {
                        format!("(uint){addr}_ok.x | ((uint){addr}_ok.y << 1)")
                    } else {
                        format!("(uint){addr}_ok")
                    };
                    let raw0 = if meshlet {
                        format!("{addr}_v.x")
                    } else {
                        format!("{addr}_v")
                    };
                    let raw1 = if meshlet {
                        format!("{addr}_v.y")
                    } else {
                        "0u".into()
                    };
                    let high = format!(
                        "(s33 & 0xFFFFu) + (uint)({addr}_a < s32) + (uint)((uint){addr}_off < {voff})"
                    );
                    add(
                        end + 1,
                        at(
                            if meshlet { 2 } else { 5 },
                            &[
                                &format!("(uint){addr}_act"),
                                &bounds,
                                &raw0,
                                &raw1,
                                voff,
                                &format!("{addr}_a & ~3u"),
                                &high,
                                window,
                                "s32",
                                "s33",
                                "s34",
                                "s35",
                            ],
                        ),
                    );
                }
                2 => add(line, at(3, &["(uint)xcl_on", "v1", "v2"])),
                3 | 9 => {}
                4 => {
                    let (end, _) = lowered()
                        .find(|&(_, text)| temp(text, "s53 = ", ";").is_some())
                        .ok_or("mesh producer cannot locate the second readfirstlane assignment")?;
                    add(end + 1, at(4, &["s52", "s53", "v25", "v3"]));
                }
                6 => add(line, at(6, &["(uint)xcl_on", "v2"])),
                7 => add(line, at(7, &["s52", "s53", "v25", "v3"])),
                8 => add(
                    line,
                    at(
                        8,
                        &[
                            "m0",
                            "m0 & 0x3FFu",
                            "(m0 >> 12) & 0x7FFu",
                            "(uint)((m0 & 0x3FFu) == 1u && ((m0 >> 12) & 0x7FFu) == 1u)",
                            "s52",
                            "s53",
                            "(uint)xcl_on",
                        ],
                    ),
                ),
                10 => {
                    add(
                        line,
                        row(
                            9,
                            self.0[9],
                            &[
                                "(uint)xcl_on",
                                "v1",
                                "0x80000000u",
                                "(uint)(v1 == 0x80000000u)",
                            ],
                        ),
                    );
                    add(
                        line,
                        at(
                            10,
                            &[
                                "(uint)xcl_on",
                                "0u",
                                "1u",
                                "v0",
                                "v0",
                                "v0",
                                "v0",
                                "xcl_vtx * 4u",
                            ],
                        ),
                    );
                }
                11 | 12 => add(
                    line,
                    at(
                        i,
                        &[
                            "(uint)xcl_on",
                            "1u",
                            "1u",
                            "v1",
                            "v0",
                            "v0",
                            "v0",
                            "xcl_prm * 4u",
                        ],
                    ),
                ),
                13 => add(
                    line,
                    at(
                        13,
                        &[
                            "(uint)xcl_on",
                            "15u",
                            "1u",
                            "v67",
                            "v68",
                            "v69",
                            "v70",
                            "xcl_vtx * 4u",
                        ],
                    ),
                ),
                _ => unreachable!(),
            }
        }

        let mut hlsl = String::with_capacity(marked.len() + 8192);
        for (n, text) in lines.into_iter().enumerate() {
            if let Some(extra) = insert.remove(&n) {
                hlsl.push_str(&extra);
            }
            if !text.trim_start().starts_with(MARK) {
                hlsl.push_str(text);
            }
        }
        if !insert.is_empty() {
            return Err("mesh producer could not apply every insertion".into());
        }
        Ok(hlsl)
    }
}

fn temp<'a>(line: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    let t = line.strip_prefix(prefix)?.strip_suffix(suffix)?;
    let digits = t.strip_prefix('t')?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(t)
}

fn row(slot: usize, pc: u64, payload: &[&str]) -> String {
    let bit = 1u32 << slot;
    let base = BASE + slot * ROW_BYTES;
    let mut h = format!(
        "    if (xcl_prod_lane0 && (xcl_prod_seen & {bit}u) == 0u) {{\n        xcl_prod_seen |= {bit}u;\n"
    );
    for (i, values) in payload.chunks(4).enumerate() {
        let mut words = ["0u"; 4];
        words[..values.len()].copy_from_slice(values);
        let _ = writeln!(
            h,
            "        probe.Store4({}u, uint4({}));",
            base + (i + 1) * 16,
            words.join(", ")
        );
    }
    let _ = writeln!(
        h,
        "        probe.Store4({base}u, uint4(0x50524f44u, {pc}u, exec_lo, exec_hi));\n    }}"
    );
    h
}

pub(crate) fn compile(
    code: &[u8],
    record: &MsInterface,
    wave64: bool,
    split: bool,
    ordinary: &[u8],
) -> Result<Vec<u8>, String> {
    if dxc::in_cached_only() || ACTIVE.with(|a| a.borrow().is_some()) {
        return Err("mesh producer compilation needs an isolated blocking request".into());
    }
    let original = stage_psv(ordinary, 5)?;
    let raw = |r| [7, 0, r, r, 11, 0];
    if original.bindings != [raw(0), raw(1)] {
        return Err(format!(
            "mesh producer ordinary CS must use only raw u0/u1: {:?}",
            original.bindings
        ));
    }
    let probe = Probe(sites()?);
    let windows = mem_windows();
    let options = Options {
        mode: Mode::Mesh(record.clone()),
        wave64,
        split_wave64: split,
        mem_windows: windows,
        ..Default::default()
    };
    let program = decode_program(code, &options)?;
    probe.verify(&program)?;
    let baseline = recomp::translate_mesh(code, record.clone(), wave64, split, windows)?;
    if !baseline.complete() || !baseline.static_samplers.is_empty() {
        return Err(
            "mesh producer requires a complete ordinary translation with zero static samplers"
                .into(),
        );
    }
    let cached = dxc::cached_only(|| dxc::compile_cached(&baseline.hlsl, "main", PROFILE)).0;
    if !cached.is_ok_and(|(dxil, hit)| hit && dxil == ordinary)
        && dxc::compile(&baseline.hlsl, "main", PROFILE, &[])? != ordinary
    {
        return Err("mesh producer could not reproduce the exact ordinary CS DXIL".into());
    }
    let trace = Rc::new(Trace {
        producer: Some(probe),
        ..Default::default()
    });
    let variant = {
        ACTIVE.with(|a| a.replace(Some(Rc::clone(&trace))));
        let _scope = TraceScope;
        emit_program(&program, &options)?
    };
    if !variant.complete()
        || !variant.static_samplers.is_empty()
        || variant.packed_compute != baseline.packed_compute
    {
        return Err("mesh producer changed the ordinary lowering or static samplers".into());
    }
    let hlsl =
        trace
            .producer
            .as_ref()
            .unwrap()
            .instrument(&baseline.hlsl, &variant.hlsl, windows)?;
    let bytes = dxc::for_program("mesh producer diagnostic CS", || {
        dxc::compile(&hlsl, "main", PROFILE, &[])
    })?;
    let psv = stage_psv(&bytes, 5)?;
    if psv.bindings != [raw(0), raw(1), raw(2)]
        || psv.runtime != original.runtime
        || psv.tail != original.tail
    {
        return Err(format!(
            "mesh producer sibling must add only raw u2 and preserve PSV metadata: {:?}",
            psv.bindings
        ));
    }
    Ok(bytes)
}
