//! AST → IR (owner: the lowering subagent).
//!
//! One walk over the entry function produces its SSA form directly: every call of a user
//! function is lowered in place (DXIL has no calls to defined functions), locals, parameters
//! and `static` globals are SSA values ([`ssa`]), and only large or shared arrays are memory.
//! The submodules: [`analysis`] (what loops assign, what may be speculated), [`ssa`] (regions,
//! joins, loop phis), `expr` (expressions and places), `stmt` (statements and control flow),
//! `call` (overload resolution, inlining, atomics on shared memory).
//!
//! # Precondition: loops around lane operations
//!
//! The output matches DXC -O1 only where the exit of a loop around a wave, quad or
//! derivative operation is the same on every lane of a (console) wave: DXC threads some ways
//! out of such loops and this lowering does not (`thread`). The translator guarantees it,
//! its control flow being scalar. `XCL_DXIL_LANE_CHECK=1` checks it ([`lanes`]) and refuses
//! what it cannot prove; the translator patterns it cannot prove are `rla`'s fallback lane
//! (`select(in_range, target, WaveGetLaneIndex())`) and, in programs from before the
//! per-lane window `xcl_wv`, the memory window global `xcl_w`.
//!
//! # The seam to [`super::stage`] (graphics stages)
//!
//! Compute entry points are lowered here entirely. For every other stage the signature
//! belongs to [`super::stage`] (its contract is in that module's documentation): the
//! lowering asks [`stage::begin`] for a [`StageIo`] and stays generic. An `Input` /
//! `InOut` parameter is an SSA variable initialised by [`StageIo::input`] in the entry
//! block, an `Output` one starts undefined, and the return value and every `Output` /
//! `InOut` parameter go to [`StageIo::output`] once, in the block every `return` of the
//! entry function reaches. The object parameters (GS input arrays and streams, patches, mesh
//! output arrays) are [`VarKind::Stage`]: not values, only `v[i]` ([`StageIo::input_at`]),
//! `s.Append(x)` / `s.RestartStrip()`, and assignments to `out[i]` or a part of it
//! ([`expr::Place::Stage`] → [`StageIo::mesh_store`]). `SetMeshOutputCounts`,
//! `DispatchMesh`, `EvaluateAttribute*` and `discard` go to their hooks.
//! [`StageIo::finish`] sees the finished body. A hull shader's patch constant function
//! has no hook yet: `Error::Unsupported("stage hs")`.

mod analysis;
mod call;
mod cfg;
mod dxbreak;
mod expr;
mod handles;
mod lanes;
mod ssa;
mod stmt;
mod thread;

use super::ast::{self, qual, AttrArg, Expr, ExprId, ScalarKind, Sym, TypeId, TypeRef, Unit, VarDecl};
use super::emit::Emit;
use super::resources;
use super::stage::{self, ParamKind, StageIo, Target};
use super::types::{HTy, Sk, StructTy, Val};
use crate::consts::{OpCode, ShaderKind};
use crate::ir::{Module, Ty, Value};
use crate::shader::{parse_profile, Resource, ShaderInfo, StageProps};
use crate::{Error, Result};
use analysis::{Analysis, SSA_ARRAY_MAX};
use ssa::{JoinId, Ssa};
use std::collections::HashMap;

/// The stack of the lowering thread.
const STACK: usize = 512 << 20;

/// Lower `entry` of `unit` for `profile`: the module holds the entry function (and the patch
/// constant function of a hull shader) and every dx.op declaration it uses, without metadata.
pub fn lower(unit: &Unit, entry: &str, profile: &str) -> Result<(Module, ShaderInfo)> {
    // statements and expressions nest as deep as the source does, and every call is lowered
    // inside its caller: give the recursion room (the parser bounds the nesting inside one
    // function; `inline` watches what calls inside calls add up to)
    std::thread::scope(|s| {
        let t = std::thread::Builder::new().stack_size(STACK).spawn_scoped(s, || Lower::new(unit).run(entry, profile));
        match t {
            Ok(t) => t.join().unwrap_or_else(|_| Err(Error::Internal("the lowering panicked".into()))),
            Err(e) => Err(Error::Internal(format!("cannot start the lowering thread: {e}"))),
        }
    })
}

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone)]
enum VarKind {
    /// first slot; the type's components follow
    Ssa(u32),
    /// index in `Lower::mems`
    Mem(u32),
    /// a register-bound resource: index in `Lower::res`
    Res(u32),
    /// an entry parameter that is an object of the stage (an input array or patch, a
    /// stream, a mesh output array): its index. The variable's type is the element type.
    Stage(u32),
}

struct Var {
    name: Sym,
    /// the variable of the same name this one hides
    prev: u32,
    /// 0: a global; else the call it is a local of
    frame: u32,
    ty: HTy,
    kind: VarKind,
}

/// An array in memory, as a flat array of one scalar type (bool is i32, as in DXC).
struct MemVar {
    name: Sym,
    /// NONE until the first use (unused globals are not emitted)
    ptr: Value,
    elem: Sk,
    n: u32,
    /// 3: groupshared
    space: u32,
    /// constant initialiser, one value per scalar
    init: Option<Vec<Value>>,
    constant: bool,
    /// a global variable of the module, else an alloca
    global: bool,
    /// not an array in the source: the global is one scalar, as DXC declares it
    scalar: bool,
}

/// What a resource variable (a local or a parameter of a resource or sampler type) holds.
/// A handle may not go through a phi or a select, so the variable is not a handle: its SSA
/// slot is the heap index (an i32 that merges like any other), and the handle is fetched
/// again where the variable is read. The value numbering makes every read that an equal
/// fetch dominates that fetch, so a handle the source hoists is fetched once.
#[derive(Copy, Clone, PartialEq, Eq)]
enum ResState {
    /// an element of a descriptor heap: the `createHandleFromHeap` flags (i1 constants)
    /// and the `annotateHandle` properties it was fetched with
    Heap { sampler: Value, nonuniform: Value, props: Value },
    /// the handle of a register-bound resource (it is in the entry block); the slot is unused
    Fixed(Value),
}

struct RetCtx {
    join: JoinId,
    slot: u32,
    ty: HTy,
    /// region depth of the function body: a `return` there with no other return before it
    /// needs no join
    depth: usize,
    tail: bool,
}

pub(super) struct Lower<'a> {
    u: &'a Unit,
    e: Emit,
    ssa: Ssa,
    an: Analysis,
    vars: Vec<Var>,
    sym_top: Vec<u32>,
    frame: u32,
    next_frame: u32,
    /// the address of a local of `run`: how much stack is in use is the distance from it
    stack_base: usize,
    mems: Vec<MemVar>,
    /// the register-bound resources declared, their types, and their handles (NONE until
    /// the first use: DXC drops unused resources)
    res: Vec<Resource>,
    res_ty: Vec<HTy>,
    res_handles: Vec<Value>,
    by_name: Vec<(u32, u32)>,
    by_name_list: Vec<u32>,
    fn_busy: Vec<bool>,
    struct_of: HashMap<Sym, u32>,
    ty_cache: Vec<Option<HTy>>,
    /// the code being lowered is unreachable (after a return, break, continue)
    dead: bool,
    breaks: Vec<JoinId>,
    continues: Vec<JoinId>,
    rets: Vec<RetCtx>,
    handle_ty: Ty,
    /// instructions spliced into the entry block so far (resource handles)
    prologue: usize,
    tmp_tys: Vec<Ty>,
    tmp_slots: Vec<u32>,
    sym_stamp: Vec<u32>,
    stamp_id: u32,
    /// the entry's inputs and outputs (None in a compute shader)
    io: Option<StageIo>,
    /// the `Input` parameters: first slot, slots, parameter (`EvaluateAttribute*` takes
    /// a part of one)
    stage_inputs: Vec<(u32, u32, u32)>,
    /// the `Output` / `InOut` parameters: variable, parameter
    stage_outs: Vec<(u32, usize)>,
    /// by slot: what the resource variable there holds, from its first assignment on
    res_state: HashMap<u32, ResState>,
    /// the preheader, latch and exit block of every loop: kept even when empty
    loop_blocks: Vec<u32>,
    /// the exit blocks among them
    loop_exits: Vec<u32>,
    /// DXC's conditional breaks: see [`dxbreak`]
    dxb: dxbreak::State,
    /// the variable the statement just lowered assigned, while [`Emit::track`] is set
    last_store: Option<Sym>,
    /// the statements after the one being lowered in its block, and how many of them it
    /// lowered itself (see `Lower::join_then`)
    rest: &'a [ast::StmtId],
    took: usize,
    /// while a statement is the last of a branch of an `if`: the join the branch ends at
    tail: Option<thread::Tail>,
    /// the loops of the function being lowered that we are in
    loops: Vec<thread::LoopFrame>,
    tmp_class: Vec<u8>,
    /// by symbol: 1 to 6 for `select`, `asfloat`, `asuint`, `asint`, `and`, `or` (see
    /// `hot_intrinsic`), else 0
    hot: Vec<u8>,
    /// emptied argument lists of built-in calls, for the next call (see `builtin_args`)
    arg_pool: Vec<(Vec<Option<expr::Place>>, Vec<Val>)>,
}

impl<'a> Lower<'a> {
    fn new(u: &'a Unit) -> Lower<'a> {
        let nsyms = u.syms.len();
        // functions by name, in declaration order (a definition replaces its prototype)
        let mut lists: HashMap<Sym, Vec<u32>> = HashMap::new();
        for (i, f) in u.funcs.iter().enumerate() {
            if f.body.is_some() {
                lists.entry(f.name).or_default().push(i as u32);
            }
        }
        let mut by_name = vec![(0u32, 0u32); nsyms];
        let mut by_name_list = Vec::new();
        for (s, l) in lists {
            by_name[s.0 as usize] = (by_name_list.len() as u32, l.len() as u32);
            by_name_list.extend(l);
        }
        let mut is_global = vec![false; nsyms];
        for g in &u.globals {
            if let ast::Global::Var(i) = g {
                is_global[u.vars[*i as usize].name.0 as usize] = true;
            }
        }
        let an = Analysis::new(u, by_name.clone(), by_name_list.clone(), is_global);
        let mut e = Emit::new();
        let handle_ty = crate::dxop::handle_ty(&mut e.m);
        Lower {
            u,
            e,
            ssa: Ssa::new(),
            an,
            vars: Vec::new(),
            sym_top: vec![NONE; nsyms],
            frame: 0,
            next_frame: 1,
            stack_base: 0,
            mems: Vec::new(),
            res: Vec::new(),
            res_ty: Vec::new(),
            res_handles: Vec::new(),
            by_name,
            by_name_list,
            fn_busy: vec![false; u.funcs.len()],
            struct_of: HashMap::new(),
            ty_cache: vec![None; u.types.len()],
            dead: false,
            breaks: Vec::new(),
            continues: Vec::new(),
            rets: Vec::new(),
            handle_ty,
            prologue: 0,
            tmp_tys: Vec::new(),
            tmp_slots: Vec::new(),
            sym_stamp: vec![0; nsyms],
            stamp_id: 0,
            io: None,
            stage_inputs: Vec::new(),
            stage_outs: Vec::new(),
            res_state: HashMap::new(),
            loop_blocks: Vec::new(),
            loop_exits: Vec::new(),
            dxb: dxbreak::State::default(),
            last_store: None,
            rest: &[],
            took: 0,
            tail: None,
            loops: Vec::new(),
            tmp_class: Vec::new(),
            arg_pool: Vec::new(),
            hot: (0..nsyms as u32)
                .map(|i| match u.name(Sym(i)) {
                    "select" => 1,
                    "asfloat" => 2,
                    "asuint" => 3,
                    "asint" => 4,
                    "and" => 5,
                    "or" => 6,
                    _ => 0,
                })
                .collect(),
        }
    }

    fn err<T>(&self, msg: impl Into<String>) -> Result<T> {
        self.e.err(msg)
    }
    fn unsupported<T>(&self, what: impl Into<String>) -> Result<T> {
        Err(Error::Unsupported(what.into()))
    }
    fn name(&self, s: Sym) -> &'a str {
        self.u.name(s)
    }

    fn run(mut self, entry: &str, profile: &str) -> Result<(Module, ShaderInfo)> {
        let here = 0u8;
        self.stack_base = std::ptr::addr_of!(here) as usize;
        let Some((kind, major, minor)) = parse_profile(profile) else {
            return Err(Error::Source { line: 0, msg: format!("unknown profile {profile}") });
        };
        let u = self.u;
        let Some(f) = u.funcs.iter().rposition(|f| f.body.is_some() && u.name(f.name) == entry) else {
            return Err(Error::Source { line: 0, msg: format!("no entry function {entry}") });
        };
        let func = &u.funcs[f];
        if let Some(sites) = self.an.lanes.half_report(u, entry) {
            return Err(Error::Unsupported(sites));
        }
        self.dxb.on = matches!(kind, ShaderKind::Pixel | ShaderKind::Compute) && self.an.any_wave();
        self.dxb.fn_lex = self.an.fn_wave_lex(f as u32);
        // DXC: "attribute WaveSize only valid for compute and node shaders"
        if kind != ShaderKind::Compute && func.attrs.iter().any(|a| u.name(a.name).eq_ignore_ascii_case("wavesize")) {
            return Err(Error::Source { line: func.line, msg: "attribute WaveSize is only valid for a compute shader".into() });
        }
        if (major, minor) != (6, 6) {
            return self.unsupported(format!("profile {profile}"));
        }
        let mut info = ShaderInfo::new(kind, entry, StageProps::Vertex);
        info.sm = (major, minor);
        if kind == ShaderKind::Hull {
            return self.unsupported("stage hs");
        }
        for g in &u.globals {
            match *g {
                ast::Global::Struct(i) => self.declare_struct(i)?,
                ast::Global::Var(i) => {
                    let d = &u.vars[i as usize];
                    self.e.line = d.line;
                    self.declare(d, true)?;
                }
                ast::Global::Func(_) => {}
            }
        }
        self.e.line = func.line;
        self.frame = self.next_frame;
        self.next_frame += 1;
        if kind == ShaderKind::Compute {
            let (mut numthreads, mut wave_size) = (None, None);
            for a in &func.attrs {
                let ints: Vec<u32> = a.args.iter().filter_map(|x| if let AttrArg::Int(v) = x { Some(*v as u32) } else { None }).collect();
                match u.name(a.name).to_ascii_lowercase().as_str() {
                    "numthreads" if ints.len() == 3 => numthreads = Some([ints[0], ints[1], ints[2]]),
                    "wavesize" if ints.len() == 1 => wave_size = Some(ints[0]),
                    other => return self.unsupported(format!("entry attribute {other}")),
                }
            }
            let Some(numthreads) = numthreads else { return self.err("a compute shader needs [numthreads]") };
            // DXC ignores an attribute outside these limits and then has none
            let [x, y, z] = numthreads.map(u64::from);
            if x == 0 || y == 0 || z == 0 || x > 1024 || y > 1024 || z > 64 || x * y * z > 1024 {
                return self.err("compute entry point must have a valid numthreads attribute");
            }
            if wave_size.is_some_and(|w| !w.is_power_of_two() || !(4..=128).contains(&w)) {
                return self.err("WaveSize must be a power of 2 between 4 and 128");
            }
            info.props = StageProps::Compute { numthreads, wave_size };
            if !matches!(self.hty(func.ret)?, HTy::Void) {
                return self.err("a compute entry function returns void");
            }
            // the entry parameters: system values
            for p in &func.params {
                let ty = self.hty(p.ty)?;
                let HTy::Num(k, 1, n) = ty else { return self.err("entry parameter type") };
                let sem = p.semantic.map(|s| u.name(s).to_ascii_lowercase()).unwrap_or_default();
                let (op, comps) = match sem.as_str() {
                    "sv_groupid" => (OpCode::GroupId, 3),
                    "sv_groupthreadid" => (OpCode::ThreadIdInGroup, 3),
                    "sv_dispatchthreadid" => (OpCode::ThreadId, 3),
                    "sv_groupindex" => (OpCode::FlattenedThreadIdInGroup, 1),
                    other => return self.unsupported(format!("compute input semantic {other}")),
                };
                if n as usize > comps || p.quals & qual::OUT != 0 || !p.dims.is_empty() {
                    return self.err("entry parameter type");
                }
                let mut c = [Value::NONE; 4];
                for (i, c) in c.iter_mut().enumerate().take(n as usize) {
                    let v = if comps == 1 {
                        self.e.dx(op, Ty::I32, &[])
                    } else {
                        let k = self.e.c_u32(i as u32);
                        self.e.dx(op, Ty::I32, &[k])
                    };
                    *c = self.e.convert_scalar(v, Sk::Uint, k);
                }
                self.new_ssa_var(p.name, ty, &c[..n as usize]);
            }
        } else {
            self.stage_params(func, kind)?;
        }
        let ret_ty = self.hty(func.ret)?;
        let ret = self.blank(ret_ty, false)?;
        let ret_slot = self.ssa.len();
        for &c in ret.comps() {
            let t = self.e.ty_of(c);
            self.ssa.alloc(t, c);
        }
        let join = self.ssa.new_join(&self.e);
        self.ssa.open_region(&self.e);
        self.rets.push(RetCtx { join, slot: ret_slot, ty: ret_ty, depth: self.ssa.region_depth(), tail: false });
        if let Some(body) = func.body {
            self.stmts(u.block(body), None)?;
        }
        if self.finish_body(join)? {
            // the one exit: DXC stores the outputs once too, before its single return
            if let Some(io) = self.io.as_mut() {
                if ret_ty != HTy::Void {
                    let n = self.e.types.comps(ret_ty);
                    // an output that was never assigned stays undefined: the stage decides (DXC
                    // does not store it and its signature element is unused)
                    let comps = self.ssa.cur[ret_slot as usize..ret_slot as usize + n].to_vec();
                    io.output(&mut self.e, Target::Return, &comps)?;
                }
                for &(var, param) in &self.stage_outs {
                    let v = &self.vars[var as usize];
                    let VarKind::Ssa(base) = v.kind else { continue };
                    let n = self.e.types.comps(v.ty);
                    let comps = self.ssa.cur[base as usize..base as usize + n].to_vec();
                    io.output(&mut self.e, Target::Param(param), &comps)?;
                }
            }
            self.e.body.ret(self.e.bb);
        }
        if self.ssa.violation {
            return self.unsupported("an assignment in a loop that the loop analysis did not see");
        }
        // DXC numbers the resources it keeps (the used ones) per class in declaration order
        let mut ids = [0u32; 5];
        for (r, h) in self.res.iter_mut().zip(&self.res_handles) {
            if !h.is_none() {
                let id = &mut ids[(r.class as u32 as usize).min(4)];
                r.id = *id;
                *id += 1;
                info.resources.push(r.clone());
            }
        }
        info.groupshared_bytes = self.mems.iter().filter(|m| m.space == 3 && !m.ptr.is_none()).map(|m| m.n * if m.elem.is_64() { 8 } else { 4 }).sum();
        let name = self.name(func.name);
        let io = self.io.take();
        let mut e = self.e;
        dxbreak::finish(&mut e, self.dxb);
        cfg::apply_forwarding(&mut e.body);
        cfg::define_reads(&mut e.body, &mut e.m);
        if cfg::thread_empty_blocks(&mut e.body, &self.loop_blocks, &self.loop_exits, self.handle_ty) {
            return Err(Error::Unsupported("a resource chosen by control flow".into()));
        }
        cfg::simplify(&mut e.body, &e.m, &self.loop_blocks, &self.loop_exits, self.handle_ty);
        handles::hoist_heap_handles(&mut e.body, &e.m);
        if let Some(io) = io {
            io.finish(&e.m, &e.body, &mut info)?;
        }
        if e.deriv_ops != 0 {
            stage::derivative_group(&e.m, &e.body, &info.props)?;
        }
        let body = std::mem::take(&mut e.body);
        e.m.define(name, body);
        Ok((e.m, info))
    }

    /// The parameters of a graphics entry function, from [`stage::begin`].
    fn stage_params(&mut self, func: &'a ast::Func, kind: ShaderKind) -> Result<()> {
        let mut io = stage::begin(&mut self.e, self.u, func, kind)?;
        if io.params().len() != func.params.len() {
            return Err(Error::Internal("stage parameters".into()));
        }
        for (i, p) in func.params.iter().enumerate() {
            let sp = io.params()[i];
            let ty = self.hty(sp.elem)?;
            self.e.line = p.line;
            match sp.kind {
                ParamKind::Input | ParamKind::InOut => {
                    let comps = io.input(&mut self.e, i)?;
                    if comps.len() != self.e.types.comps(ty) {
                        return Err(Error::Internal("stage input components".into()));
                    }
                    let base = self.ssa.len();
                    let var = self.new_ssa_var(p.name, ty, &comps);
                    if sp.kind == ParamKind::Input {
                        self.stage_inputs.push((base, comps.len() as u32, i as u32));
                    } else {
                        self.stage_outs.push((var, i));
                    }
                }
                ParamKind::Output => {
                    let v = self.blank(ty, false)?;
                    let var = self.new_ssa_var(p.name, ty, v.comps());
                    self.stage_outs.push((var, i));
                }
                _ => {
                    self.push_var(p.name, ty, VarKind::Stage(i as u32));
                }
            }
        }
        self.io = Some(io);
        Ok(())
    }

    /// The end of a function body whose returns go to `join`: true when the code after it is
    /// reachable.
    fn finish_body(&mut self, join: JoinId) -> Result<bool> {
        let Some(ctx) = self.rets.pop() else { return Err(Error::Internal("return context".into())) };
        if !self.ssa.has_edges(join) && (!self.dead || ctx.tail) {
            self.ssa.commit_region();
            self.ssa.drop_join(join);
            self.dead = false;
            return Ok(true);
        }
        if !self.dead {
            self.ssa.add_edge(join, self.e.bb, ssa::Term::Br);
        }
        self.ssa.undo_region(&mut self.e);
        self.dead = !self.ssa.resolve(&mut self.e, join);
        Ok(!self.dead)
    }

    // ---- types ----

    fn declare_struct(&mut self, i: u32) -> Result<()> {
        let s = &self.u.structs[i as usize];
        let mut fields = Vec::with_capacity(s.fields.len());
        for f in &s.fields {
            let ty = self.decl_ty(f)?;
            fields.push((self.name(f.name).to_string(), ty));
        }
        self.struct_of.insert(s.name, self.e.types.structs.len() as u32);
        self.e.types.structs.push(StructTy { name: self.name(s.name).to_string(), fields });
        Ok(())
    }

    fn hty(&mut self, t: TypeId) -> Result<HTy> {
        if let Some(h) = self.ty_cache[t.0 as usize] {
            return Ok(h);
        }
        let h = match *self.u.ty(t) {
            TypeRef::Void => HTy::Void,
            TypeRef::Num { kind, rows, cols } => {
                let k = match kind {
                    ScalarKind::Bool => Sk::Bool,
                    ScalarKind::Int | ScalarKind::Min16Int => Sk::Int,
                    ScalarKind::Uint | ScalarKind::Min16Uint => Sk::Uint,
                    ScalarKind::Int64 => Sk::Int64,
                    ScalarKind::Uint64 => Sk::Uint64,
                    ScalarKind::Half | ScalarKind::Float | ScalarKind::Min16Float => Sk::Float,
                    ScalarKind::Double => Sk::Double,
                    ScalarKind::Int16 | ScalarKind::Uint16 | ScalarKind::Float16 => return self.unsupported("16-bit types"),
                };
                if rows > 1 {
                    return self.unsupported("matrix types");
                }
                HTy::Num(k, rows, cols)
            }
            TypeRef::Named(s) => match self.struct_of.get(&s) {
                Some(&i) => HTy::Struct(i),
                None => return self.err(format!("unknown type {}", self.name(s))),
            },
            TypeRef::Resource { kind, elem, samples } => {
                let elem = match elem {
                    Some(t) => Some(self.hty(t)?),
                    None => None,
                };
                HTy::Res(resources::res_ty(kind, elem, samples, false, &self.e.types)?)
            }
            TypeRef::Sampler { comparison } => HTy::Sampler { comparison },
            TypeRef::Stream { .. } | TypeRef::Patch { .. } => return self.unsupported("stream and patch types"),
        };
        self.ty_cache[t.0 as usize] = Some(h);
        Ok(h)
    }

    /// The value of a constant integer expression (array sizes, case labels).
    fn const_int(&mut self, x: ExprId) -> Result<u64> {
        let v = self.expr(x)?;
        match (v.ty, v.len()) {
            (HTy::Num(k, 1, 1), 1) if k.is_int() || k == Sk::Bool => match self.e.m.as_int(v.one()) {
                Some(c) => Ok(c),
                None => self.err("a constant integer expression is expected"),
            },
            _ => self.err("a constant integer expression is expected"),
        }
    }

    /// The type of a declaration: its written type, its array dimensions, and the qualifiers
    /// that are part of a resource type.
    fn decl_ty(&mut self, d: &VarDecl) -> Result<HTy> {
        let mut ty = self.hty(d.ty)?;
        if let HTy::Res(r) = &mut ty {
            r.coherent = d.quals & qual::GLOBALLYCOHERENT != 0;
        }
        for i in (0..d.dims.len()).rev() {
            let n = self.const_int(d.dims[i])?;
            if n == 0 || n > 1 << 24 {
                return self.err("array size");
            }
            ty = self.e.types.array(ty, n as u32);
        }
        Ok(ty)
    }

    // ---- variables ----

    fn push_var(&mut self, name: Sym, ty: HTy, kind: VarKind) -> u32 {
        let i = self.vars.len() as u32;
        let prev = std::mem::replace(&mut self.sym_top[name.0 as usize], i);
        self.vars.push(Var { name, prev, frame: self.frame, ty, kind });
        i
    }
    fn new_ssa_var(&mut self, name: Sym, ty: HTy, comps: &[Value]) -> u32 {
        let base = self.ssa.len();
        for &c in comps {
            let t = self.e.ty_of(c);
            self.ssa.alloc(t, c);
        }
        self.push_var(name, ty, VarKind::Ssa(base))
    }
    /// A resource variable (see [`ResState`]), assigned the handle `init` if there is one.
    fn new_res_var(&mut self, name: Sym, ty: HTy, init: Option<Value>) -> Result<u32> {
        let undef = self.e.undef(Ty::I32);
        let slot = self.ssa.len();
        let var = self.new_ssa_var(name, ty, &[undef]);
        // slots are reused once a scope ends
        self.res_state.remove(&slot);
        if let Some(h) = init {
            self.res_var_store(slot, h)?;
        }
        Ok(var)
    }

    /// The operands of `v` after the opcode, if it is a call of the dx.op `op`.
    fn dx_operands(&self, v: Value, op: OpCode) -> Option<&[Value]> {
        let x = self.e.def(self.e.body.resolve(v))?;
        if x.op != crate::ir::Op::Call {
            return None;
        }
        let args = self.e.body.operands.get(x.b.0 as usize..(x.b.0 + x.c.0) as usize)?;
        (self.e.m.as_int(*args.first()?) == Some(op as u64)).then(|| &args[1..])
    }

    /// Assign the annotated handle `h` to the resource variable at `slot`: what the
    /// handle was fetched from is what the variable holds. Every assignment of one variable
    /// must agree on everything but the heap index.
    pub(super) fn res_var_store(&mut self, slot: u32, h: Value) -> Result<()> {
        let fetch = self.dx_operands(h, OpCode::AnnotateHandle).map(|a| (a[0], a[1]));
        let (state, index) = match fetch {
            Some((raw, props)) => match self.dx_operands(raw, OpCode::CreateHandleFromHeap) {
                Some(c) => (ResState::Heap { sampler: c[1], nonuniform: c[2], props }, Some(c[0])),
                None if self.dx_operands(raw, OpCode::CreateHandleFromBinding).is_some() && self.res_handles.contains(&h) => (ResState::Fixed(h), None),
                None => return self.unsupported("a resource variable assigned something other than a heap element or a bound resource"),
            },
            None => return self.unsupported("a resource variable assigned something other than a heap element or a bound resource"),
        };
        match self.res_state.insert(slot, state) {
            Some(old) if old != state => return self.unsupported("a resource variable assigned resources of different kinds"),
            _ => {}
        }
        if let Some(index) = index {
            self.ssa.write(slot, index);
        }
        Ok(())
    }

    /// The handle of the resource variable at `slot`, fetched here (or the equal fetch
    /// that dominates this point).
    pub(super) fn res_var_load(&mut self, slot: u32, ty: HTy) -> Result<Val> {
        match self.res_state.get(&slot).copied() {
            None => self.err("a resource variable is used before it is assigned"),
            Some(ResState::Fixed(h)) => Ok(Val::new(ty, &[h])),
            Some(ResState::Heap { sampler, nonuniform, props }) => {
                let index = self.ssa.read(slot);
                let h = self.e.dx(OpCode::CreateHandleFromHeap, Ty::VOID, &[index, sampler, nonuniform]);
                let h = self.e.dx(OpCode::AnnotateHandle, Ty::VOID, &[h, props]);
                Ok(Val::new(ty, &[h]))
            }
        }
    }

    /// The variable `name` means here: a local of the function being lowered, or a global.
    fn lookup(&self, name: Sym) -> Option<u32> {
        let mut i = self.sym_top[name.0 as usize];
        while i != NONE {
            let v = &self.vars[i as usize];
            if v.frame == self.frame || v.frame == 0 {
                return Some(i);
            }
            i = v.prev;
        }
        None
    }
    fn scope(&self) -> (usize, u32) {
        (self.vars.len(), self.ssa.len())
    }
    fn end_scope(&mut self, mark: (usize, u32)) {
        while self.vars.len() > mark.0 {
            if let Some(v) = self.vars.pop() {
                self.sym_top[v.name.0 as usize] = v.prev;
            }
        }
        self.ssa.truncate(mark.1);
    }

    /// The undefined (or, for `zero`, zero) value of a type, as SSA components.
    fn blank(&mut self, ty: HTy, zero: bool) -> Result<Val> {
        let mut tys = std::mem::take(&mut self.tmp_tys);
        tys.clear();
        self.e.types.comp_tys(ty, self.handle_ty, &mut tys);
        let n = tys.len();
        let mut few = [Value::NONE; 8];
        let mut many = Vec::new();
        for (i, &t) in tys.iter().enumerate() {
            let c = if zero && t != self.handle_ty { self.e.zero_of(t) } else { self.e.undef(t) };
            if n <= 8 {
                few[i] = c;
            } else {
                many.push(c);
            }
        }
        self.tmp_tys = tys;
        Ok(if n <= 8 { Val::new(ty, &few[..n]) } else { Val::new(ty, &many) })
    }

    /// The scalar kinds of a numeric aggregate.
    fn sks(&mut self, ty: HTy) -> Result<Vec<Sk>> {
        let mut out = Vec::new();
        if !self.e.types.comp_sks(ty, &mut out) {
            return self.err("an aggregate of numeric types is expected");
        }
        Ok(out)
    }

    /// Flatten an initialiser (`{ a, { b, c } }` or an expression) to the scalars of `sks`.
    fn init_scalars(&mut self, x: ExprId, sks: &[Sk], out: &mut Vec<Value>) -> Result<()> {
        if let Expr::InitList(r) = *self.u.expr(x) {
            for &a in self.u.args(r) {
                self.init_scalars(a, sks, out)?;
            }
            return Ok(());
        }
        let v = self.expr(x)?;
        let mut from = Vec::new();
        if !self.e.types.comp_sks(v.ty, &mut from) {
            return self.err("initialiser type");
        }
        for (i, &c) in v.comps().iter().enumerate() {
            let Some(&to) = sks.get(out.len()) else { return self.err("too many initialisers") };
            out.push(self.e.convert_scalar(c, from[i], to));
        }
        Ok(())
    }

    /// The initial value of a variable of type `ty` from its initialiser.
    fn init_value(&mut self, x: ExprId, ty: HTy) -> Result<Val> {
        if let Expr::InitList(_) = *self.u.expr(x) {
            let sks = self.sks(ty)?;
            let mut out = Vec::with_capacity(sks.len());
            self.init_scalars(x, &sks, &mut out)?;
            if out.len() != sks.len() {
                return self.err("too few initialisers");
            }
            return Ok(Val::new(ty, &out));
        }
        let v = self.expr(x)?;
        self.coerce(&v, ty)
    }

    fn declare(&mut self, d: &VarDecl, global: bool) -> Result<()> {
        let ty = self.decl_ty(d)?;
        let is_static = d.quals & qual::STATIC != 0;
        let shared = d.quals & qual::GROUPSHARED != 0;
        if let (true, HTy::Res(_) | HTy::Sampler { .. }) = (global && !is_static, ty) {
            let name = self.name(d.name);
            let i = resources::declare(&mut self.res, name, ty, d.register.as_ref())?;
            self.res_ty.push(ty);
            self.res_handles.push(Value::NONE);
            self.push_var(d.name, ty, VarKind::Res(i as u32));
            return Ok(());
        }
        if global && !is_static && !shared {
            return self.unsupported("a global outside static / groupshared / resources (constant buffers)");
        }
        if let HTy::Res(_) | HTy::Sampler { .. } = ty {
            if is_static || shared {
                return self.unsupported("static resource variables");
            }
            let init = match d.init {
                Some(x) => {
                    let v = self.expr(x)?;
                    Some(self.coerce(&v, ty)?.one())
                }
                None => None,
            };
            self.new_res_var(d.name, ty, init)?;
            return Ok(());
        }
        if let HTy::Array(_) = ty {
            let n = self.e.types.comps(ty) as u64;
            let (_, count) = self.e.types.elem(ty).unwrap_or((ty, 0));
            let konst = d.quals & qual::CONST != 0 && is_static && d.init.is_some();
            if shared || konst || n > 4 * SSA_ARRAY_MAX || count as u64 > SSA_ARRAY_MAX {
                return self.declare_mem(d, ty, global || is_static, shared, konst);
            }
        } else if shared {
            return self.declare_mem(d, ty, true, true, false);
        }
        let v = match d.init {
            Some(x) => self.init_value(x, ty)?,
            None => self.blank(ty, global || is_static)?,
        };
        if matches!(ty, HTy::ResHeap | HTy::SamplerHeap | HTy::HeapElem { .. } | HTy::Void) {
            return self.err("variable type");
        }
        // a static local keeps its value between calls in HLSL; nothing the translator emits
        if is_static && !global {
            return self.unsupported("static locals");
        }
        self.new_ssa_var(d.name, ty, v.comps());
        Ok(())
    }

    fn declare_mem(&mut self, d: &VarDecl, ty: HTy, is_static: bool, shared: bool, konst: bool) -> Result<()> {
        let sks = self.sks(ty)?;
        let elem = sks[0].concrete();
        if sks.iter().any(|k| k.concrete() != elem) {
            return self.unsupported("an array in memory with elements of several types");
        }
        let n = sks.len() as u32;
        let mut init = None;
        if let Some(x) = d.init {
            let mut out = Vec::with_capacity(sks.len());
            self.init_scalars(x, &sks, &mut out)?;
            if out.len() != sks.len() {
                return self.err("too few initialisers");
            }
            init = Some(out);
        }
        if shared && init.is_some() {
            return self.err("a groupshared variable has no initialiser");
        }
        let all_const = init.as_ref().is_some_and(|v| v.iter().all(|c| c.is_const()));
        let i = self.mems.len() as u32;
        self.mems.push(MemVar { name: d.name, ptr: Value::NONE, elem, n, space: if shared { 3 } else { 0 }, init: None, constant: false, global: is_static || shared, scalar: n == 1 && !matches!(ty, HTy::Array(_)) });
        self.push_var(d.name, ty, VarKind::Mem(i));
        if is_static || shared {
            if init.is_some() && !all_const {
                return self.unsupported("a static array with a computed initialiser");
            }
            // booleans are i32 in memory
            if let (Some(v), Sk::Bool) = (&mut init, elem) {
                for c in v.iter_mut() {
                    *c = self.e.cast(crate::ir::CastOp::ZExt, Ty::I32, *c);
                }
            }
            self.mems[i as usize].init = init;
            self.mems[i as usize].constant = konst;
        } else {
            // a local array: an alloca, initialised here
            let ptr = self.mem_ptr(i);
            if let Some(v) = init {
                for (k, &c) in v.iter().enumerate() {
                    let idx = self.e.c_u32(k as u32);
                    self.mem_store(i, ptr, idx, c);
                }
            }
        }
        Ok(())
    }

    fn mem_ir(elem: Sk) -> Ty {
        if elem == Sk::Bool {
            Ty::I32
        } else {
            elem.ir()
        }
    }

    /// The pointer of an array in memory, created at its first use: an internal global (a
    /// constant one for a `static const` table), an external global in address space 3 for
    /// `groupshared`, as DXC emits them; an alloca for a local.
    fn mem_ptr(&mut self, i: u32) -> Value {
        let mv = &self.mems[i as usize];
        if !mv.ptr.is_none() {
            return mv.ptr;
        }
        let et = Self::mem_ir(mv.elem);
        let (n, space, name) = (mv.n, mv.space, self.name(mv.name));
        let scalar = mv.scalar;
        let at = if scalar { et } else { self.e.m.array(n as u64, et) };
        let is_global = mv.global;
        let ptr = if space == 3 {
            // DXC's name for it (the MS mangling of a global array), where it is simple
            let code = match mv.elem {
                Sk::Uint => Some("I"),
                Sk::Int => Some("H"),
                Sk::Float => Some("M"),
                _ => None,
            };
            let mangled = match code {
                Some(c) if scalar => format!("\u{1}?{name}@@3{c}A"),
                Some(c) => format!("\u{1}?{name}@@3PA{c}A"),
                None => name.to_string(),
            };
            let g = self.e.m.global(&mangled, at, 3, None, false);
            if let crate::ir::ValueKind::Global(k) = g.kind() {
                self.e.m.globals[k as usize].internal = false;
            }
            g
        } else if is_global {
            let init = match &self.mems[i as usize].init {
                Some(v) => {
                    let v = v.clone();
                    self.e.m.aggregate(at, v)
                }
                None => self.e.zero_of(at),
            };
            let constant = self.mems[i as usize].constant;
            self.e.m.global(name, at, 0, Some(init), constant)
        } else {
            let pt = self.e.m.ptr(at);
            self.e.body.alloca(pt)
        };
        self.mems[i as usize].ptr = ptr;
        ptr
    }

    fn mem_elem_ptr(&mut self, i: u32, ptr: Value, index: Value) -> Value {
        let mv = &self.mems[i as usize];
        if mv.scalar {
            return ptr;
        }
        let (et, space) = (Self::mem_ir(mv.elem), mv.space);
        let pt = self.e.m.ptr_in(et, space);
        let zero = self.e.c_u32(0);
        self.e.touch(index);
        let p = self.e.body.gep(self.e.bb, pt, ptr, &[zero, index]);
        // DXC leaves inbounds off where a run-time index addresses a groupshared array
        if space == 3 && self.e.m.as_int(index).is_none() {
            self.e.body.insts[p.0 as usize].flags |= crate::ir::flag::NO_INBOUNDS;
        }
        p
    }
    fn mem_store(&mut self, i: u32, ptr: Value, index: Value, v: Value) {
        let p = self.mem_elem_ptr(i, ptr, index);
        let v = if self.mems[i as usize].elem == Sk::Bool { self.e.cast(crate::ir::CastOp::ZExt, Ty::I32, v) } else { v };
        self.e.touch(v);
        self.e.body.store(self.e.bb, p, v);
    }
    fn mem_load(&mut self, i: u32, index: Value) -> Value {
        // a constant index into a constant table is the constant
        if let (Some(k), true) = (self.e.m.as_int(index), self.mems[i as usize].constant) {
            if let Some(&c) = self.mems[i as usize].init.as_ref().and_then(|v| v.get(k as usize)) {
                return if self.mems[i as usize].elem == Sk::Bool { let z = self.e.c_u32(0); self.e.icmp(crate::ir::IPred::Ne, c, z) } else { c };
            }
        }
        let ptr = self.mem_ptr(i);
        let p = self.mem_elem_ptr(i, ptr, index);
        let mv = &self.mems[i as usize];
        let (elem, table, scalar) = (mv.elem, mv.constant && mv.global && mv.space != 3, mv.scalar);
        let mut v = self.e.body.load(self.e.bb, Self::mem_ir(elem), p);
        if table {
            v = self.e.number_table_load(v, ptr, if scalar { Value::NONE } else { index });
        }
        if elem == Sk::Bool {
            let z = self.e.c_u32(0);
            self.e.icmp(crate::ir::IPred::Ne, v, z)
        } else {
            v
        }
    }

    /// The annotated handle of a register-bound resource, created once, at the start of the
    /// entry block (where DXC creates the handles of bound resources).
    fn res_handle(&mut self, i: u32) -> Result<Value> {
        let h = self.res_handles[i as usize];
        if !h.is_none() {
            return Ok(h);
        }
        let blocks = self.e.body.blocks.len();
        let stage = self.e.body.new_block();
        let (bb, mark) = (std::mem::replace(&mut self.e.bb, stage), self.e.vn_mark());
        let r = match self.res_ty[i as usize] {
            HTy::Res(rt) => resources::bound_handle(&mut self.e, &self.res[i as usize], &rt),
            _ => resources::bound_sampler(&mut self.e, &self.res[i as usize]),
        };
        self.e.vn_release(mark);
        self.e.bb = bb;
        if self.e.body.blocks.len() != blocks + 1 {
            return Err(Error::Internal("bound_handle created blocks".into()));
        }
        let staged = self.e.body.blocks.pop().map(|b| b.insts).unwrap_or_default();
        let h = r?;
        let entry = &mut self.e.body.blocks[0].insts;
        let insts = &self.e.body.insts;
        let allocas = entry.iter().position(|&k| insts[k as usize].op != crate::ir::Op::Alloca).unwrap_or(entry.len());
        let at = (allocas + self.prologue).min(entry.len());
        self.prologue += staged.len();
        entry.splice(at..at, staged);
        self.res_handles[i as usize] = h;
        Ok(h)
    }

    /// Convert a value to the type of what it is assigned or passed to.
    fn coerce(&mut self, v: &Val, to: HTy) -> Result<Val> {
        if v.ty == to {
            return Ok(v.clone());
        }
        match (v.ty, to) {
            (HTy::Num(..), HTy::Num(..)) => self.e.convert(v, to),
            (HTy::HeapElem { sampler: false, nonuniform }, HTy::Res(rt)) => {
                let h = resources::heap_handle(&mut self.e, v.one(), nonuniform, Some(&rt), false)?;
                Ok(Val::new(to, &[h]))
            }
            // the handle keeps the annotation it was fetched with (DXC warns that
            // globallycoherent is lost and passes the handle on)
            (HTy::Res(a), HTy::Res(b)) if a.kind == b.kind && a.uav == b.uav && a.elem == b.elem && a.elem_comps == b.elem_comps => Ok(Val::new(to, v.comps())),
            (HTy::HeapElem { sampler: true, nonuniform }, HTy::Sampler { comparison }) => {
                let h = resources::heap_handle(&mut self.e, v.one(), nonuniform, None, comparison)?;
                Ok(Val::new(to, &[h]))
            }
            _ => self.err(format!("cannot convert {:?} to {:?}", v.ty, to)),
        }
    }
}
