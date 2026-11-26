use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

use crate::gpu::rdna2::Decoded;
use crate::gpu::shader::fnv64;

use super::super::{decode_program, emit_program, loop_profile};
use super::{dxc, legacy_compute_selected, mem_windows, with_console_code, Compiled, Stage};

pub const MAGIC: u32 = 0x4741_5445;
pub(crate) mod producer;
mod source_depth;

pub const BYTES: usize = 196;

#[derive(Clone, Copy)]
struct Points {
    pre: u64,
    vgpr: u8,
    post: u64,
    store: u64,
}

struct Selection {
    hash: u64,
    quadrants: Vec<Points>,
    stores: Vec<u64>,
}

fn hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s.trim().trim_start_matches("0x"), 16)
        .map_err(|_| format!("expected a hex shader hash or PC, got {s:?}"))
}

fn selection() -> Option<&'static Selection> {
    static SELECTION: OnceLock<Option<Selection>> = OnceLock::new();
    SELECTION
        .get_or_init(|| {
            let hash = std::env::var_os("XCL_MATERIAL_GATE_CS")?;
            let parse = || -> Result<Selection, String> {
                let hash = hex(hash.to_str().ok_or("invalid XCL_MATERIAL_GATE_CS")?)?;
                if let Ok(sites) = std::env::var("XCL_WRITER_STORE_PCS") {
                    let stores = sites.split(',').map(hex).collect::<Result<Vec<_>, _>>()?;
                    if stores.len() != 8 || stores.iter().collect::<std::collections::BTreeSet<_>>().len() != 8 {
                        return Err("expected eight distinct writer store PCs".into());
                    }
                    return Ok(Selection { hash, quadrants: Vec::new(), stores });
                }
                let config = std::env::var("XCL_MATERIAL_GATE_POINTS")
                    .map_err(|_| "XCL_MATERIAL_GATE_POINTS is required")?;
                let mut quadrants = Vec::new();
                for fields in config.split(',') {
                    let fields: Vec<_> = fields.split(':').map(str::trim).collect();
                    let [pre, vgpr, post, store] = fields[..] else {
                        return Err("each quadrant needs pre_pc:vgpr:post_execz_pc:store_pc".into());
                    };
                    let p = Points {
                        pre: hex(pre)?,
                        vgpr: vgpr
                            .trim_start_matches('v')
                            .parse()
                            .map_err(|_| "expected a VGPR (decimal)")?,
                        post: hex(post)?,
                        store: hex(store)?,
                    };
                    if p.pre >= p.post || p.post >= p.store {
                        return Err("quadrant PCs must be in pre, post, store order".into());
                    }
                    quadrants.push(p);
                }
                let quadrants: [Points; 4] = quadrants
                    .try_into()
                    .map_err(|_| "expected four quadrants")?;
                let mut pcs = std::collections::BTreeSet::new();
                if !quadrants
                    .iter()
                    .flat_map(|p| [p.pre, p.post, p.store])
                    .all(|pc| pcs.insert(pc))
                {
                    return Err("quadrant PCs must be distinct".into());
                }
                Ok(Selection { hash, quadrants: quadrants.to_vec(), stores: Vec::new() })
            };
            match parse() {
                Ok(config) => Some(config),
                Err(e) => {
                    crate::xlog_warn!("[material-gate] probe disabled: {e}");
                    None
                }
            }
        })
        .as_ref()
}

pub fn selected_hash() -> Option<u64> {
    selection().map(|s| s.hash)
}

pub fn writer_mode() -> bool { selection().is_some_and(|s| !s.stores.is_empty()) }

pub(crate) fn stage_bindings(dxil: &[u8], stage: u8) -> Result<Vec<[u32; 6]>, &'static str> {
    Ok(stage_psv(dxil, stage)?.bindings)
}

struct StagePsv<'a> {
    runtime: &'a [u8],
    tail: &'a [u8],
    bindings: Vec<[u32; 6]>,
}

fn stage_psv(dxil: &[u8], stage: u8) -> Result<StagePsv<'_>, &'static str> {
    let parts = crate::gpu::shader::container_parts(dxil).ok_or("invalid DXBC container")?;
    let mut psv = parts.iter().filter(|part| part.fourcc == "PSV0");
    let part = psv.next().ok_or("no PSV0")?;
    if psv.next().is_some() { return Err("duplicate PSV0") }
    let p = &dxil[part.offset..part.offset + part.size];
    let mut bytes = p;
    let word = |r: &mut &[u8]| -> Result<u32, &'static str> {
        let (w, rest) = r.split_first_chunk::<4>().ok_or("truncated PSV0")?;
        *r = rest;
        Ok(u32::from_le_bytes(*w))
    };
    let size = word(&mut bytes)? as usize;
    if !matches!(size, 36 | 48 | 52) { return Err("unsupported PSV0 runtime-info version") }
    let (runtime, rest) = bytes.split_at_checked(size).ok_or("truncated PSV0 runtime info")?;
    if runtime[24] != stage { return Err("PSV0 shader stage mismatch") }
    bytes = rest;
    let n = word(&mut bytes)? as usize;
    if n == 0 { return Ok(StagePsv { runtime, tail: bytes, bindings: Vec::new() }) }
    if word(&mut bytes)? != 24 { return Err("unsupported PSV0 binding-record size") }
    let (table, tail) = bytes.split_at_checked(n.checked_mul(24).ok_or("invalid PSV0 binding count")?).ok_or("truncated PSV0 binding table")?;
    let mut entries = Vec::with_capacity(n);
    for mut row in table.chunks_exact(24) {
        let mut entry = [0; 6];
        for value in &mut entry { *value = word(&mut row)?; }
        if !(1..=9).contains(&entry[0]) || entry[3] < entry[2] { return Err("unknown PSV0 binding or invalid register range") }
        entries.push(entry);
    }
    entries.sort_unstable();
    Ok(StagePsv { runtime, tail, bindings: entries })
}

pub fn compile(blob: &[u8], split_wave64: bool, ordinary: &Compiled) -> Result<Compiled, String> {
    let hash = fnv64(blob);
    let config = selection()
        .filter(|s| s.hash == hash)
        .ok_or("shader is not selected")?;
    if dxc::in_cached_only() {
        return Err("material-gate compilation requires a blocking request".into());
    }
    if loop_profile::enabled_for(blob) {
        return Err("material-gate does not support loop profiling on the same shader".into());
    }
    with_console_code(
        blob,
        Stage::Compute,
        0,
        mem_windows(),
        split_wave64,
        |code, mut options, profile| {
            options.legacy_compute_wave64 |= split_wave64 && legacy_compute_selected(hash);
            let program = decode_program(code, &options)?;
            let mut baseline = emit_program(&program, &options)?;
            let same = |hlsl: &str| match dxc::cached_only(|| dxc::compile_cached(hlsl, "main", profile)).0 {
                Ok((dxil, true)) => dxil == ordinary.dxil,
                _ => dxc::compile(hlsl, "main", profile, &[]).is_ok_and(|dxil| dxil == ordinary.dxil),
            };
            if !same(&baseline.hlsl) {
                if !baseline.packed_compute { return Err("material-gate cannot verify the ordinary cached DXIL".into()) }
                options.legacy_compute_wave64 = true;
                baseline = emit_program(&program, &options)?;
                if !same(&baseline.hlsl) { return Err("material-gate cannot verify either ordinary wave64 lowering".into()) }
            }
            if !baseline.complete() { return Err("material-gate needs a complete ordinary translation".into()) }
            if baseline.static_samplers != ordinary.samplers {
                return Err(format!("material-gate static samplers differ: regenerated={:?} live={:?}", baseline.static_samplers, ordinary.samplers));
            }
            let mut expected = stage_bindings(&ordinary.dxil, 5).map_err(|why| format!("material-gate cannot verify live UAV bindings: {why}"))?;
            if let Some(range) = expected.iter().find(|&r| (6..=9).contains(&r[0]) && r[1] == 0 && r[2] <= 1 && 1 <= r[3]) {
                return Err(format!("material-gate live shader uses u1, space0: PSV0 {range:?}"));
            }

            let trace = Rc::new(Trace {
                producer: None,
                source: std::env::var("XCL_MATERIAL_GATE_SOURCE_PC").ok().map(|pc| source_depth::Site::new(hex(&pc)?, &program, &options)).transpose()?,
                source_seen: Cell::new(false),
                quadrants: config.quadrants.clone(),
                stores: config.stores.iter().map(|pc| {
                    let d = program.insts.iter().find(|d| d.pc == *pc).ok_or("missing writer image store")?;
                    let f = |name| crate::gpu::rdna2::field_value(d.enc, &d.words, name);
                    if d.inst.name != "IMAGE_STORE" || f("DIM") != Some(1) || f("D16") != Some(0) || f("DMASK").is_none_or(|m| m & 1 == 0) {
                        return Err("writer probe needs ordinary 2D image stores with a red channel");
                    }
                    Ok((*pc, f("VDATA").ok_or("missing writer image data")?))
                }).collect::<Result<Vec<_>, &str>>()?,
                seen: Cell::new(0),
                failed: Cell::new(false),
            });
            let variant = {
                ACTIVE.with(|a| a.replace(Some(trace.clone())));
                let _scope = TraceScope;
                emit_program(&program, &options)?
            };
            if trace.source.is_some() != trace.source_seen.get() { return Err("source-depth site was not emitted".into()) }
            let expected_sites = if trace.stores.is_empty() { 0xfff } else { 0xff };
            if !trace.stores.is_empty() && trace.source.is_none() { return Err("missing writer input gather PC".into()) }
            if trace.failed.get() || trace.seen.get() != expected_sites {
                return Err(format!("material-gate sites do not match this lowering (seen={:#x} expected={expected_sites:#x})", trace.seen.get()));
            }
            if !variant.complete()
                || variant.packed_compute != baseline.packed_compute
                || variant.static_samplers != ordinary.samplers
            {
                return Err("material-gate changed the shader lowering or static samplers".into());
            }
            let dxil = dxc::for_program(&format!("material-gate CS {hash:016x}"), || {
                dxc::compile(&variant.hlsl, "main", profile, &[])
            })?;
            expected.push([7, 0, 1, 1, 11, 0]);
            expected.sort_unstable();
            let actual = stage_bindings(&dxil, 5).map_err(|why| format!("material-gate cannot verify variant bindings: {why}"))?;
            if actual != expected {
                return Err(format!("material-gate variant must add only the raw result UAV u1, space0: expected={expected:?} actual={actual:?}"));
            }
            Ok(Compiled {
                dxil,
                samplers: variant.static_samplers,
            })
        },
    )
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<Trace>>> = const { RefCell::new(None) };
}

struct TraceScope;

impl Drop for TraceScope {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.replace(None));
    }
}

pub(in crate::gpu::recomp) fn current() -> Option<Rc<Trace>> {
    ACTIVE.with(|a| a.borrow().clone())
}

#[derive(Default)]
pub(in crate::gpu::recomp) struct Trace {
    producer: Option<producer::Probe>,
    source: Option<source_depth::Site>,
    source_seen: Cell<bool>,
    quadrants: Vec<Points>,
    stores: Vec<(u64, u64)>,
    seen: Cell<u16>,
    failed: Cell<bool>,
}

impl Trace {
    fn mark(&self, quadrant: usize, stage: usize) -> usize {
        let i = quadrant * 3 + stage;
        let bit = 1 << i;
        if self.seen.get() & bit != 0 {
            self.failed.set(true);
        }
        self.seen.set(self.seen.get() | bit);
        20 + 4 * i
    }

    pub(in crate::gpu::recomp) fn before(&self, d: &Decoded, packed: bool) -> Option<String> {
        if let Some(probe) = &self.producer {
            return probe.before(d, packed);
        }
        if let Some(site) = self.source.as_ref().filter(|site| site.after == d.pc) {
            if self.source_seen.replace(true) { self.failed.set(true); }
            return Some(site.emit(packed, !self.stores.is_empty()));
        }
        for (q, p) in self.quadrants.iter().enumerate() {
            let pre = d.pc == p.pre;
            if !pre && d.pc != p.post {
                continue;
            }
            if !pre && d.inst.name != "S_CBRANCH_EXECZ" {
                self.failed.set(true);
                return None;
            }
            let at = self.mark(q, if pre { 0 } else { 1 });
            let tap = |prefix: &str| {
                if pre {
                    let bits = format!("{prefix}v{}", p.vgpr);
                    format!(" && {bits} != 0u && {bits} <= 0x7F800000u")
                } else {
                    String::new()
                }
            };
            let mut hlsl = format!("if (xcl_gate_on && xcl_on{}) {{ uint xcl_gate_old; res.InterlockedAdd({at}u, 1u, xcl_gate_old); }}", tap(""));
            if packed {
                hlsl.push_str(&format!(" if (xcl_gate_on && hi_xcl_on{}) {{ uint xcl_gate_old; res.InterlockedAdd({at}u, 1u, xcl_gate_old); }}", tap("hi_")));
            }
            return Some(hlsl);
        }
        None
    }

    pub(in crate::gpu::recomp) fn store(
        &self,
        pc: u64,
        dim: u64,
        index: &str,
        coordinate: &str,
        active: &str,
    ) -> Option<String> {
        if let Some((n, &(_, data))) = self.stores.iter().enumerate().find(|(_, s)| s.0 == pc) {
            let bit = 1 << n;
            if dim != 1 || self.seen.get() & bit != 0 { self.failed.set(true); return None; }
            self.seen.set(self.seen.get() | bit);
            let at = 4 * (if n < 4 { 5 + 3 * n } else { 49 + 3 * (n - 4) });
            return Some(format!("if (xcl_gate_on && {active}) {{ uint old; uint2 xy = (uint2)({coordinate}); bool ok = {index} + 1u == res.Load(4u) && all(xy < uint2(res.Load(8u),res.Load(12u))); uint mag = v{data} & 0x7fffffffu; uint a = WaveActiveCountBits(true), b = WaveActiveCountBits(ok), c = WaveActiveCountBits(ok && mag != 0u && mag <= 0x7f800000u), nanbits = WaveActiveCountBits(ok && mag > 0x7f800000u); if (WaveIsFirstLane()) {{ res.InterlockedAdd({at}u, a, old); res.InterlockedAdd({}u, b, old); res.InterlockedAdd({}u, c, old); res.InterlockedAdd(244u, nanbits, old); }} if (ok && all(xy == uint2(res.Load(8u), res.Load(12u)) / 2u)) {{ res.InterlockedAdd(248u, 1u, old); res.InterlockedOr(252u, v{data}, old); }} }}", at + 4, at + 8));
        }
        let q = self.quadrants.iter().position(|p| p.store == pc)?;
        if dim != 1 {
            self.failed.set(true);
            return None;
        }
        let at = self.mark(q, 2);
        Some(format!(
            "if (xcl_gate_on && {active}) {{ uint2 xcl_gate_xy = (uint2)({coordinate}); if ({index} + 1u == res.Load(4u) && xcl_gate_xy.x < res.Load(8u) && xcl_gate_xy.y < res.Load(12u)) {{ uint xcl_gate_old; res.InterlockedAdd({at}u, 1u, xcl_gate_old); }} }}"
        ))
    }
}

pub(in crate::gpu::recomp) const ENTRY: &str = concat!(
    "    bool xcl_gate_on = res.Load(0u) == 0x47415445u;\n",
    "    if (xcl_gate_on && gi == 0u && all(gid == uint3(0u, 0u, 0u))) { uint xcl_gate_old; res.InterlockedAdd(16u, 1u, xcl_gate_old); }\n",
);
