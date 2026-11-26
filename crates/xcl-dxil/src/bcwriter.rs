//! [`crate::ir::Module`] → LLVM 3.7 bitcode (owner: the bitcode subagent).
//!
//! The layout is the one DXC's own writer produces (lib/Bitcode/Writer, which is LLVM 3.7's):
//! module version 1 (operands relative to the instruction's value id), the BLOCKINFO
//! abbreviations LLVM defines (load, binop, cast, ret, unreachable, gep in function blocks;
//! settype, integer, null in constants; the symbol table entries), attribute groups, the type
//! table, globals and function records, module constants, the metadata block and the
//! metadata kinds, the symbol table sorted by name (DXC's change), then the function blocks
//! with their `!dx.precise` attachments.
//!
//! Where this writer differs from DXC's, by choice (the reader takes both, see the tests):
//! every constant is in the module-level constants block (LLVM puts the ones instructions
//! use in a block per function; the value ids, and so the size, come out the same), the
//! type table holds only what is used, and label / metadata types are left out.
//!
//! What is written is what is live: per function, forwarding is resolved, blocks unreachable
//! from the entry are dropped, and only instructions with [`flag::EFFECT`] and what they use
//! (through operands and the phi incomings of reachable predecessors) stay. Constants,
//! globals and declarations appear only when something written refers to them.

use crate::bitstream::{is_char6, AbbrevOp, Writer};
use crate::ir::{flag, BinOp, Body, CastOp, Const, FnAttrs, Inst, MdNode, Module, Op, Sealed, Ty, TypeDef, Value};
use crate::{Error, Result};
use std::collections::HashMap;

#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    /// live instructions written
    pub instructions: usize,
    pub blocks: usize,
}

/// Block ids and record codes of LLVM 3.7 bitcode (`llvm/Bitcode/LLVMBitCodes.h`).
pub mod code {
    pub const MODULE_BLOCK: u32 = 8;
    pub const PARAMATTR_BLOCK: u32 = 9;
    pub const PARAMATTR_GROUP_BLOCK: u32 = 10;
    pub const CONSTANTS_BLOCK: u32 = 11;
    pub const FUNCTION_BLOCK: u32 = 12;
    pub const VALUE_SYMTAB_BLOCK: u32 = 14;
    pub const METADATA_BLOCK: u32 = 15;
    pub const METADATA_ATTACHMENT_BLOCK: u32 = 16;
    pub const TYPE_BLOCK_NEW: u32 = 17;
    pub const USELIST_BLOCK: u32 = 18;

    pub const MODULE_VERSION: u32 = 1;
    pub const MODULE_TRIPLE: u32 = 2;
    pub const MODULE_DATALAYOUT: u32 = 3;
    pub const MODULE_GLOBALVAR: u32 = 7;
    pub const MODULE_FUNCTION: u32 = 8;

    pub const PARAMATTR_ENTRY: u32 = 2;
    pub const PARAMATTR_GRP_ENTRY: u32 = 3;
    pub const ATTR_NO_CAPTURE: u64 = 11;
    pub const ATTR_NO_DUPLICATE: u64 = 12;
    pub const ATTR_NO_RETURN: u64 = 17;
    pub const ATTR_NO_UNWIND: u64 = 18;
    pub const ATTR_READ_NONE: u64 = 20;
    pub const ATTR_READ_ONLY: u64 = 21;

    pub const TYPE_NUMENTRY: u32 = 1;
    pub const TYPE_VOID: u32 = 2;
    pub const TYPE_FLOAT: u32 = 3;
    pub const TYPE_DOUBLE: u32 = 4;
    pub const TYPE_LABEL: u32 = 5;
    pub const TYPE_OPAQUE: u32 = 6;
    pub const TYPE_INTEGER: u32 = 7;
    pub const TYPE_POINTER: u32 = 8;
    pub const TYPE_HALF: u32 = 10;
    pub const TYPE_ARRAY: u32 = 11;
    pub const TYPE_VECTOR: u32 = 12;
    pub const TYPE_METADATA: u32 = 16;
    pub const TYPE_STRUCT_ANON: u32 = 18;
    pub const TYPE_STRUCT_NAME: u32 = 19;
    pub const TYPE_STRUCT_NAMED: u32 = 20;
    pub const TYPE_FUNCTION: u32 = 21;

    pub const CST_SETTYPE: u32 = 1;
    pub const CST_NULL: u32 = 2;
    pub const CST_UNDEF: u32 = 3;
    pub const CST_INTEGER: u32 = 4;
    pub const CST_FLOAT: u32 = 6;
    pub const CST_AGGREGATE: u32 = 7;
    pub const CST_CE_CAST: u32 = 11;
    pub const CAST_BITCAST: u32 = 11;
    pub const CST_CE_GEP: u32 = 12;
    pub const CST_CE_INBOUNDS_GEP: u32 = 20;
    pub const CST_STRING: u32 = 8;
    pub const CST_CSTRING: u32 = 9;
    pub const CST_DATA: u32 = 22;

    pub const METADATA_STRING: u32 = 1;
    pub const METADATA_VALUE: u32 = 2;
    pub const METADATA_NODE: u32 = 3;
    pub const METADATA_NAME: u32 = 4;
    pub const METADATA_KIND: u32 = 6;
    pub const METADATA_NAMED_NODE: u32 = 10;
    pub const METADATA_ATTACHMENT: u32 = 11;

    pub const VST_ENTRY: u32 = 1;
    pub const VST_BBENTRY: u32 = 2;

    pub const FUNC_DECLAREBLOCKS: u32 = 1;
    pub const FUNC_BINOP: u32 = 2;
    pub const FUNC_CAST: u32 = 3;
    pub const FUNC_RET: u32 = 10;
    pub const FUNC_BR: u32 = 11;
    pub const FUNC_SWITCH: u32 = 12;
    pub const FUNC_UNREACHABLE: u32 = 15;
    pub const FUNC_PHI: u32 = 16;
    pub const FUNC_ALLOCA: u32 = 19;
    pub const FUNC_LOAD: u32 = 20;
    pub const FUNC_EXTRACTVAL: u32 = 26;
    pub const FUNC_CMP2: u32 = 28;
    pub const FUNC_VSELECT: u32 = 29;
    pub const FUNC_CALL: u32 = 34;
    pub const FUNC_ATOMICRMW: u32 = 38;
    pub const FUNC_GEP: u32 = 43;
    pub const FUNC_STORE: u32 = 44;
    pub const FUNC_CMPXCHG: u32 = 46;

    /// The metadata kinds DXC's context has registered when it writes a shader module, in
    /// its order: LLVM 3.7's fixed kinds, then DXC's own.
    pub const MD_KINDS: [&str; 20] = [
        "dbg",
        "tbaa",
        "prof",
        "fpmath",
        "range",
        "tbaa.struct",
        "invariant.load",
        "alias.scope",
        "noalias",
        "nontemporal",
        "llvm.mem.parallel_loop_access",
        "nonnull",
        "dereferenceable",
        "dereferenceable_or_null",
        "dx.temp",
        "dx.break.br",
        "dx.dbg.varlayout",
        "dx.precise",
        "dx.controlflow.hints",
        "llvm.loop",
    ];
    pub const MD_KIND_PRECISE: u64 = 17;
    pub const MD_KIND_HINTS: u64 = 18;
    pub const MD_KIND_LOOP: u64 = 19;
    pub const METADATA_DISTINCT_NODE: u32 = 5;
}
use code::*;

// abbreviation ids BLOCKINFO gives each block kind (LLVM's WriteBlockInfo)
const VST_ENTRY_8: u32 = 4;
const VST_ENTRY_7: u32 = 5;
const VST_ENTRY_6: u32 = 6;
const CST_SETTYPE_ABBREV: u32 = 4;
const CST_INTEGER_ABBREV: u32 = 5;
const FN_LOAD_ABBREV: u32 = 4;
const FN_BINOP_ABBREV: u32 = 5;
const FN_BINOP_FLAGS_ABBREV: u32 = 6;
const FN_CAST_ABBREV: u32 = 7;
const FN_RET_VOID_ABBREV: u32 = 8;
const FN_UNREACHABLE_ABBREV: u32 = 10;
const FN_GEP_ABBREV: u32 = 11;

const NONE: u32 = u32::MAX;

fn internal<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Internal(msg.into()))
}

/// What of a body gets written: the blocks reachable from the entry and, in those, the
/// instructions with an effect plus everything they use (after forwarding).
#[derive(Debug, Default, Clone)]
pub struct Live {
    /// per block
    pub blocks: Vec<bool>,
    /// per instruction
    pub insts: Vec<bool>,
}

/// Call f with each successor of the terminator t (targets outside the body are skipped).
pub(crate) fn successors(body: &Body, t: &Inst, mut f: impl FnMut(u32)) {
    let n = body.blocks.len() as u32;
    let mut go = |b: u32| {
        if b < n {
            f(b)
        }
    };
    match t.op {
        Op::Br => go(t.a.0),
        Op::CondBr => {
            go(t.b.0);
            go(t.c.0);
        }
        Op::Switch => {
            go(t.b.0);
            if let Some(cases) = body.switches.get(t.c.0 as usize) {
                for &(_, b) in cases {
                    go(b.0);
                }
            }
        }
        _ => {}
    }
}

/// Call f with each value operand of an instruction as it is stored (not resolved). Phi
/// incomings count only from blocks `reachable` marks.
#[inline(always)]
fn operands(body: &Body, inst: &Inst, reachable: &[bool], mut f: impl FnMut(Value)) {
    match inst.op {
        Op::Bin(_) | Op::ICmp(_) | Op::FCmp(_) | Op::Store | Op::AtomicRmw(_) => {
            f(inst.a);
            f(inst.b);
        }
        Op::Cast(_) | Op::Load | Op::ExtractValue | Op::CondBr | Op::Switch => f(inst.a),
        Op::Select | Op::CmpXchg => {
            f(inst.a);
            f(inst.b);
            f(inst.c);
        }
        Op::Call | Op::Gep => {
            f(inst.a);
            let (first, n) = (inst.b.0 as usize, inst.c.0 as usize);
            if let Some(ops) = body.operands.get(first..first.saturating_add(n)) {
                for &v in ops {
                    f(v);
                }
            }
        }
        Op::Phi => {
            if let Some(incoming) = body.phis.get(inst.a.0 as usize) {
                for &(v, from) in incoming {
                    if reachable.get(from.0 as usize).copied().unwrap_or(false) {
                        f(v);
                    }
                }
            }
        }
        Op::Alloca | Op::Br | Op::Ret | Op::Unreachable => {}
    }
}

pub(crate) fn reachable_blocks(body: &Body, work: &mut Vec<u32>) -> Vec<bool> {
    let mut blocks = vec![false; body.blocks.len()];
    if blocks.is_empty() {
        return blocks;
    }
    blocks[0] = true;
    work.push(0);
    while let Some(b) = work.pop() {
        let Some(t) = body.blocks[b as usize].insts.last().and_then(|&i| body.insts.get(i as usize)) else { continue };
        successors(body, t, |s| {
            if !blocks[s as usize] {
                blocks[s as usize] = true;
                work.push(s);
            }
        });
    }
    blocks
}

/// The liveness of a body, operands read through `resolve`.
///
/// An instruction is created after its operands, except for the incoming values of a phi
/// (added once their blocks exist) and for what forwarding points at. So one sweep from the
/// last instruction to the first settles everything else in order, without a worklist: when
/// the sweep reaches an instruction, every later user has marked it already. Only an operand
/// created after its user goes through the worklist.
#[inline(always)]
fn live_with(body: &Body, resolve: impl Fn(Value) -> Value) -> Live {
    let mut work: Vec<u32> = Vec::new();
    let blocks = reachable_blocks(body, &mut work);
    let n = body.insts.len();
    // which instructions are in a reachable block: the block lists alone, so that the
    // instructions themselves are read once, in the sweep, in the order they are stored
    let mut placed = vec![false; n];
    for (b, data) in body.blocks.iter().enumerate() {
        if !blocks[b] {
            continue;
        }
        for list in [&data.phis, &data.insts] {
            for &i in list {
                if let Some(p) = placed.get_mut(i as usize) {
                    *p = true;
                }
            }
        }
    }
    let mut insts = vec![false; n];
    for i in (0..n).rev() {
        if !insts[i] {
            if !(placed[i] && body.insts[i].flags & flag::EFFECT != 0) {
                continue;
            }
            insts[i] = true;
        }
        let mut at = i;
        loop {
            operands(body, &body.insts[at], &blocks, |v| {
                let v = resolve(v);
                let k = v.0 as usize;
                if v.is_inst() && k < n && !insts[k] {
                    insts[k] = true;
                    if k > i {
                        work.push(v.0);
                    }
                }
            });
            match work.pop() {
                Some(next) => at = next as usize,
                None => break,
            }
        }
    }
    Live { blocks, insts }
}

/// The liveness the writer applies (see the module text), for [`crate::irprint`] and tests.
/// Never panics on a malformed body.
pub fn liveness(body: &Body) -> Live {
    if body.has_replacements() {
        live_with(body, |v| body.resolve(v))
    } else {
        live_with(body, |v| v)
    }
}

/// [`liveness`] for the writer: a forwarding cycle or a forwarding of something that does not
/// exist is an error here (and would not end there). `name`: the function, for the error text.
pub(crate) fn checked_liveness(body: &Body, name: &str) -> Result<Live> {
    let fwd = forwarding(body, name)?;
    if fwd.is_empty() {
        return Ok(live_with(body, |v| v));
    }
    Ok(live_with(body, |v| if v.is_inst() { fwd.get(v.0 as usize).map_or(v, |&to| Value(to)) } else { v }))
}

/// The liveness of every defined function, by function index: what [`write`] computes for a
/// module that is not sealed ([`crate::ir::Sealed`] holds the same).
pub(crate) fn module_liveness(m: &Module) -> Result<Vec<Option<Live>>> {
    m.funcs.iter().map(|f| f.body.as_ref().map(|b| checked_liveness(b, &f.name)).transpose()).collect()
}

/// The alignment DXC gives loads, stores, allocas and globals of a type: 8 when it holds a
/// 64-bit scalar, else 4 (the DXIL data layout aligns everything narrower to 32 bits).
pub fn align_of(m: &Module, t: Ty) -> u32 {
    match m.types.get(t.0 as usize) {
        Some(TypeDef::Int(64)) | Some(TypeDef::Double) => 8,
        Some(TypeDef::Array { of, .. }) | Some(TypeDef::Vector { of, .. }) => align_of(m, *of),
        Some(TypeDef::Struct { fields, .. }) => fields.iter().map(|&f| align_of(m, f)).max().unwrap_or(4),
        _ => 4,
    }
}

/// LLVM's `Log2_32_Ceil`.
fn log2_ceil(n: u32) -> u32 {
    32 - n.saturating_sub(1).leading_zeros()
}

/// LLVM's alignment field: log2 + 1, 0 for none.
fn align_field(align: u32) -> u32 {
    if align == 0 {
        0
    } else {
        32 - align.leading_zeros()
    }
}

/// LLVM's `emitSignedInt64`.
#[inline(always)]
fn signed_vbr(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) << 1
    } else {
        ((v as u64).wrapping_neg() << 1) | 1
    }
}

fn sext(v: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        v as i64
    } else {
        ((v << (64 - bits)) as i64) >> (64 - bits)
    }
}

/// What the module level of the output holds, collected while the function bodies are
/// analysed, then numbered.
struct Marks<'a> {
    m: &'a Module,
    types: Vec<bool>,
    consts: Vec<bool>,
    globals: Vec<bool>,
    funcs: Vec<bool>,
    /// integer constants the output needs that the module does not hold (switch case
    /// values, the `i32 1` of an alloca and of `!dx.precise`): (type, bits)
    extra: Vec<(Ty, u64)>,
    extra_at: HashMap<(Ty, u64), u32>,
    any_precise: bool,
    /// bit n: a terminator carries the control flow hints n ([`flag::HINTS`])
    hints: u64,
    /// number of terminators with [`flag::LOOP_UNROLL_DISABLE`]
    loops: u32,
    /// an operand named something that does not exist
    out_of_range: bool,
}

impl<'a> Marks<'a> {
    fn ty(&mut self, t: Ty) {
        match self.types.get_mut(t.0 as usize) {
            Some(u) => *u = true,
            None => self.out_of_range = true,
        }
    }

    fn constant(&mut self, i: u32) {
        let m = self.m;
        match self.consts.get_mut(i as usize) {
            Some(u) if *u => return,
            Some(u) => *u = true,
            None => {
                self.out_of_range = true;
                return;
            }
        }
        let c = &m.consts[i as usize];
        self.ty(c.ty);
        match &c.c {
            Const::Aggregate(fields) => {
                for &f in fields {
                    self.value(f);
                }
            }
            _ => {}
        }
    }

    /// A constant, global or function something written refers to.
    #[inline(always)]
    fn value(&mut self, v: Value) {
        let i = v.0 & 0x3fff_ffff;
        match v.0 >> 30 {
            1 => {
                if !self.consts.get(i as usize).copied().unwrap_or(false) {
                    self.constant(i)
                }
            }
            2 => match self.globals.get_mut(i as usize) {
                Some(u) => *u = true,
                None => self.out_of_range = true,
            },
            3 => match self.funcs.get_mut(i as usize) {
                Some(u) => *u = true,
                None => self.out_of_range = true,
            },
            _ => self.out_of_range = true,
        }
    }

    /// An integer constant by value: the module's own if it has it, else an extra one.
    fn int(&mut self, ty: Ty, v: u64) {
        match self.m.lookup_constant(ty, Const::Int(v)) {
            Some(c) => self.value(c),
            None => {
                self.ty(ty);
                let n = self.extra.len() as u32;
                self.extra_at.entry((ty, v)).or_insert_with(|| {
                    self.extra.push((ty, v));
                    n
                });
            }
        }
    }
}

/// One function body, analysed.
struct Plan<'a> {
    func: u32,
    body: &'a Body,
    live: &'a [bool],
    /// per block: its index among the written ones, or NONE
    block_ids: Vec<u32>,
    blocks: u32,
    /// per instruction: the value id of what it stands for (its own when written, its
    /// replacement's when forwarded), or NONE
    ids: Vec<u32>,
    /// per instruction, when the body has forwarding: the raw [`Value`] it stands for
    fwd: Vec<u32>,
    /// number of live instructions
    count: usize,
}

/// The forwarding table as a dense array, chains followed. A cycle is an error.
#[inline(never)]
fn forwarding(body: &Body, name: &str) -> Result<Vec<u32>> {
    if !body.has_replacements() {
        return Ok(Vec::new());
    }
    let n = body.insts.len();
    let mut fwd: Vec<u32> = (0..n as u32).collect();
    let mut keys = Vec::new();
    for (old, new) in body.replacements() {
        if old as usize >= n || (new.is_inst() && new.0 as usize >= n) {
            return internal(format!("{name}: forwarding of an instruction that does not exist"));
        }
        fwd[old as usize] = new.0;
        keys.push(old);
    }
    let limit = keys.len();
    for &k in &keys {
        // find the end of the chain, then point every link at it
        let mut v = fwd[k as usize];
        let mut steps = 0;
        while v >> 30 == 0 && fwd[v as usize] != v {
            v = fwd[v as usize];
            steps += 1;
            if steps > limit {
                return internal(format!("{name}: forwarding cycle at instruction {k}"));
            }
        }
        let mut at = k;
        while at != v && at >> 30 == 0 && fwd[at as usize] != at {
            let next = fwd[at as usize];
            fwd[at as usize] = v;
            at = next;
        }
    }
    Ok(fwd)
}

/// `live`: the liveness of `body` ([`checked_liveness`]). One pass over what is written, in
/// block order: the blocks get their ids, and the types, constants, globals and functions the
/// live instructions name are marked.
#[inline(never)]
fn analyse<'a>(m: &'a Module, func: u32, body: &'a Body, live: &'a Live, marks: &mut Marks<'a>) -> Result<Plan<'a>> {
    let name = &m.funcs[func as usize].name;
    let fwd = forwarding(body, name)?;
    let n = body.insts.len();
    if live.blocks.len() != body.blocks.len() || live.insts.len() != n {
        return internal(format!("{name}: the liveness is that of another body"));
    }
    let reachable = &live.blocks[..];
    let live = &live.insts[..];
    let mut block_ids = vec![NONE; body.blocks.len()];
    let mut blocks = 0u32;
    let mut count = 0usize;
    for (b, data) in body.blocks.iter().enumerate() {
        if !reachable[b] {
            continue;
        }
        block_ids[b] = blocks;
        blocks += 1;
        match data.insts.last().and_then(|&i| body.insts.get(i as usize)) {
            Some(t) if t.op.is_terminator() => {}
            _ => return internal(format!("{name}: block {b} has no terminator")),
        }
        for list in [&data.phis, &data.insts] {
            for &i in list {
                let Some(inst) = body.insts.get(i as usize) else { return internal(format!("{name}: block {b} lists an instruction that does not exist")) };
                if !live[i as usize] {
                    continue;
                }
                count += 1;
                marks.ty(inst.ty);
                operands(body, inst, reachable, |v| {
                    let v = if v.is_inst() && !fwd.is_empty() { Value(fwd.get(v.0 as usize).copied().unwrap_or(NONE)) } else { v };
                    if v.is_inst() {
                        marks.out_of_range |= v.0 as usize >= n;
                    } else {
                        marks.value(v);
                    }
                });
                match inst.op {
                    Op::Br => marks.loops += (inst.flags & flag::LOOP_UNROLL_DISABLE != 0) as u32,
                    Op::CondBr => {
                        marks.hints |= 1 << (inst.flags & flag::HINTS);
                        marks.loops += (inst.flags & flag::LOOP_UNROLL_DISABLE != 0) as u32;
                    }
                    Op::Call => marks.any_precise |= inst.flags & flag::PRECISE != 0,
                    Op::Alloca => {
                        marks.int(Ty::I32, 1);
                        marks.any_precise |= inst.flags & flag::PRECISE != 0;
                    }
                    Op::Switch => {
                        marks.hints |= 1 << (inst.flags & flag::HINTS);
                        let cond = if fwd.is_empty() || !inst.a.is_inst() { inst.a } else { Value(fwd[inst.a.0 as usize]) };
                        let Some(ty) = value_ty(m, body, cond) else { return internal(format!("{name}: switch on a value without a type")) };
                        let Some(cases) = body.switches.get(inst.c.0 as usize) else { return internal(format!("{name}: switch without a case list")) };
                        for &(v, _) in cases {
                            marks.int(ty, v);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if marks.out_of_range {
        return internal(format!("{name}: an operand names a value that does not exist"));
    }
    Ok(Plan { func, body, live, block_ids, blocks, ids: Vec::new(), fwd, count })
}

/// The type of a resolved value (None for a function, which has no interned pointer type).
fn value_ty(m: &Module, body: &Body, v: Value) -> Option<Ty> {
    let i = (v.0 & 0x3fff_ffff) as usize;
    match v.0 >> 30 {
        0 => body.insts.get(i).map(|x| x.ty),
        1 => m.consts.get(i).map(|c| c.ty),
        2 => m.globals.get(i).map(|g| g.ptr_ty),
        _ => None,
    }
}

/// A type of the output's table.
#[derive(Copy, Clone)]
enum TypeEntry {
    /// one of the module's
    Module(Ty),
    /// a pointer to a function type the module never interned (metadata names functions
    /// by value, which needs their pointer type)
    FnPtr(Ty),
}

/// A constant of the output's table.
#[derive(Copy, Clone)]
enum ConstEntry {
    Module(u32),
    Extra(u32),
}

struct Ids {
    type_ids: Vec<u32>,
    /// per function type: the id of the pointer to it, when the output has one
    fn_ptr_ids: HashMap<u32, u32>,
    type_list: Vec<TypeEntry>,
    type_bits: u32,
    global_ids: Vec<u32>,
    func_ids: Vec<u32>,
    const_ids: Vec<u32>,
    extra_ids: Vec<u32>,
    /// globals, functions and constants in id order
    globals: Vec<u32>,
    funcs: Vec<u32>,
    consts: Vec<ConstEntry>,
    /// number of module-level values: the id of the first instruction of every function
    base: u32,
    /// metadata in id order, as indices of [`Module::md`]; `md_ids` is the inverse
    md_order: Vec<u32>,
    md_ids: Vec<u32>,
    /// the id of the `!{i32 1}` node (which is written after `md_order` when the module
    /// has none), and of the `i32 1` constant it holds
    precise_node: u32,
    precise_one: u32,
    precise_synthetic: bool,
    /// The `!dx.controlflow.hints` nodes, written after those: the metadata id of the
    /// string (NONE when no terminator has a hint), of the value of each hint, of the
    /// distinct node of each set of hints a terminator has, and the constant of each hint.
    hint_string: u32,
    hint_values: [u32; 6],
    hint_nodes: [u32; 64],
    hint_consts: [u32; 6],
    /// The `!llvm.loop` nodes, written last: the metadata id of the string (NONE when there
    /// is no such loop); the node that holds it follows, then one distinct node per loop.
    loop_string: u32,
    loops: u32,
}

fn metadata_order(m: &Module, md: u32, order: &mut Vec<u32>, ids: &mut [u32], depth: u32) -> Result<()> {
    let Some(node) = m.md.get(md as usize) else { return internal("metadata operand that does not exist") };
    if ids[md as usize] != NONE {
        return Ok(());
    }
    if depth > 4096 {
        return internal("metadata nested too deeply");
    }
    if let MdNode::Tuple(ops) = node {
        for op in ops.iter().flatten() {
            metadata_order(m, op.0, order, ids, depth + 1)?;
        }
    }
    ids[md as usize] = order.len() as u32;
    order.push(md);
    Ok(())
}

/// Number everything the marks say is written.
#[inline(never)]
fn number(m: &Module, marks: &mut Marks) -> Result<Ids> {
    // metadata, in LLVM's order: operands before their node, from the named metadata
    let mut md_order = Vec::new();
    let mut md_ids = vec![NONE; m.md.len()];
    for (_, nodes) in &m.named_md {
        for n in nodes {
            metadata_order(m, n.0, &mut md_order, &mut md_ids, 0)?;
        }
    }
    let mut fn_ptr_needed: Vec<Ty> = Vec::new();
    let mut precise_node = NONE;
    for &md in &md_order {
        match &m.md[md as usize] {
            MdNode::Val(v) => {
                if v.is_inst() || v.is_none() {
                    return internal("metadata holds an instruction value");
                }
                marks.value(*v);
                match v.kind() {
                    crate::ir::ValueKind::Func(f) => match m.funcs.get(f as usize) {
                        Some(f) => fn_ptr_needed.push(f.ty),
                        None => return internal("metadata names a function that does not exist"),
                    },
                    crate::ir::ValueKind::Global(g) => {
                        if let Some(g) = m.globals.get(g as usize) {
                            marks.ty(g.ptr_ty);
                        }
                    }
                    _ => {}
                }
            }
            MdNode::Tuple(ops) if ops.len() == 1 && precise_node == NONE => {
                if let Some(MdNode::Val(v)) = ops[0].and_then(|o| m.md.get(o.0 as usize)) {
                    if m.as_const(*v).is_some_and(|c| c.ty == Ty::I32 && c.c == Const::Int(1)) {
                        precise_node = md_ids[md as usize];
                    }
                }
            }
            _ => {}
        }
    }
    let precise_synthetic = marks.any_precise && precise_node == NONE;
    if precise_synthetic {
        marks.int(Ty::I32, 1);
        precise_node = md_order.len() as u32 + 1;
    }
    let (mut hint_string, mut hint_values, mut hint_nodes) = (NONE, [NONE; 6], [NONE; 64]);
    let mut next = md_order.len() as u32 + if precise_synthetic { 2 } else { 0 };
    if marks.hints & !1 != 0 {
        hint_string = next;
        next += 1;
        let all = marks.hints;
        let sets = (1..64usize).filter(|&set| all >> set & 1 != 0);
        let used = sets.clone().fold(0, |all, set| all | set);
        for (h, id) in hint_values.iter_mut().enumerate() {
            if used >> h & 1 != 0 {
                marks.int(Ty::I32, h as u64 + 1);
                *id = next;
                next += 1;
            }
        }
        for set in sets {
            hint_nodes[set] = next;
            next += 1;
        }
    }
    let loop_string = if marks.loops != 0 { next } else { NONE };
    let loops = marks.loops;
    if marks.out_of_range {
        return internal("metadata names a value that does not exist");
    }

    // globals bring their initialiser, functions their type
    for (i, g) in m.globals.iter().enumerate() {
        if marks.globals[i] {
            marks.ty(g.ptr_ty);
            marks.ty(g.value_ty);
            if let Some(init) = g.init {
                if !init.is_const() {
                    return internal(format!("global {} has an initialiser that is not a constant", g.name));
                }
                marks.value(init);
            }
        }
    }
    // an aggregate initialiser can name further globals and functions: settle
    loop {
        let before = marks.globals.iter().filter(|&&u| u).count();
        for (i, g) in m.globals.iter().enumerate() {
            if marks.globals[i] {
                marks.ty(g.ptr_ty);
                marks.ty(g.value_ty);
                if let Some(init) = g.init {
                    marks.value(init);
                }
            }
        }
        if marks.globals.iter().filter(|&&u| u).count() == before {
            break;
        }
    }
    for (i, f) in m.funcs.iter().enumerate() {
        if marks.funcs[i] {
            if let Some(e) = crate::dxop::error_of(&f.name) {
                return Err(e);
            }
            if !matches!(m.types.get(f.ty.0 as usize), Some(TypeDef::Func { .. })) {
                return internal(format!("function {} has a type that is not a function type", f.name));
            }
            marks.ty(f.ty);
        }
    }
    if marks.out_of_range {
        return internal("a global or constant names a value or type that does not exist");
    }

    // types: components have lower handles than what is built from them, so one pass down
    // closes the set
    let mut existing_fn_ptr: HashMap<u32, u32> = HashMap::new();
    for (i, t) in m.types.iter().enumerate() {
        if let TypeDef::Ptr { to, space: 0 } = t {
            if matches!(m.types.get(to.0 as usize), Some(TypeDef::Func { .. })) {
                existing_fn_ptr.insert(to.0, i as u32);
            }
        }
    }
    for t in &fn_ptr_needed {
        if let Some(&p) = existing_fn_ptr.get(&t.0) {
            marks.types[p as usize] = true;
        }
    }
    for i in (0..m.types.len()).rev() {
        if !marks.types[i] {
            continue;
        }
        let mut need = |t: Ty| -> Result<()> {
            if t.0 as usize >= i {
                return internal("a type is built from a type with a higher handle");
            }
            marks.types[t.0 as usize] = true;
            Ok(())
        };
        match &m.types[i] {
            TypeDef::Ptr { to, .. } => need(*to)?,
            TypeDef::Array { of, .. } | TypeDef::Vector { of, .. } => need(*of)?,
            TypeDef::Struct { fields, .. } => {
                for &f in fields {
                    need(f)?;
                }
            }
            TypeDef::Func { ret, params } => {
                need(*ret)?;
                for &p in params {
                    need(p)?;
                }
            }
            _ => {}
        }
    }
    let mut type_ids = vec![NONE; m.types.len()];
    let mut type_list = Vec::new();
    let mut fn_ptr_ids = HashMap::new();
    for i in 0..m.types.len() {
        if !marks.types[i] {
            continue;
        }
        type_ids[i] = type_list.len() as u32;
        type_list.push(TypeEntry::Module(Ty(i as u32)));
        if let TypeDef::Ptr { to, space: 0 } = &m.types[i] {
            if existing_fn_ptr.get(&to.0) == Some(&(i as u32)) {
                fn_ptr_ids.insert(to.0, type_ids[i]);
            }
        }
        if fn_ptr_needed.contains(&Ty(i as u32)) && !existing_fn_ptr.contains_key(&(i as u32)) {
            fn_ptr_ids.insert(i as u32, type_list.len() as u32);
            type_list.push(TypeEntry::FnPtr(Ty(i as u32)));
        }
    }
    let type_bits = log2_ceil(type_list.len() as u32 + 1);

    // values: globals, functions, constants; each in the module's order
    let mut next = 0u32;
    let mut global_ids = vec![NONE; m.globals.len()];
    let mut globals = Vec::new();
    for i in 0..m.globals.len() {
        if marks.globals[i] {
            global_ids[i] = next;
            globals.push(i as u32);
            next += 1;
        }
    }
    let mut func_ids = vec![NONE; m.funcs.len()];
    let mut funcs = Vec::new();
    for i in 0..m.funcs.len() {
        if marks.funcs[i] {
            func_ids[i] = next;
            funcs.push(i as u32);
            next += 1;
        }
    }
    // constants grouped by type (one SETTYPE per type), in type order
    let mut per_type = vec![0u32; type_list.len() + 1];
    for (i, c) in m.consts.iter().enumerate() {
        if marks.consts[i] {
            per_type[type_ids[c.ty.0 as usize] as usize + 1] += 1;
        }
    }
    for &(ty, _) in &marks.extra {
        per_type[type_ids[ty.0 as usize] as usize + 1] += 1;
    }
    for t in 1..per_type.len() {
        per_type[t] += per_type[t - 1];
    }
    let total = per_type[type_list.len()] as usize;
    let mut consts = vec![ConstEntry::Extra(0); total];
    let mut const_ids = vec![NONE; m.consts.len()];
    let mut extra_ids = vec![NONE; marks.extra.len()];
    for (i, c) in m.consts.iter().enumerate() {
        if marks.consts[i] {
            let slot = &mut per_type[type_ids[c.ty.0 as usize] as usize];
            consts[*slot as usize] = ConstEntry::Module(i as u32);
            const_ids[i] = next + *slot;
            *slot += 1;
        }
    }
    for (i, &(ty, _)) in marks.extra.iter().enumerate() {
        let slot = &mut per_type[type_ids[ty.0 as usize] as usize];
        consts[*slot as usize] = ConstEntry::Extra(i as u32);
        extra_ids[i] = next + *slot;
        *slot += 1;
    }
    next += total as u32;

    let mut precise_one = NONE;
    if precise_synthetic {
        precise_one = match m.lookup_constant(Ty::I32, Const::Int(1)) {
            Some(c) => const_ids[(c.0 & 0x3fff_ffff) as usize],
            None => marks.extra_at.get(&(Ty::I32, 1)).map_or(NONE, |&e| extra_ids[e as usize]),
        };
    }
    let mut hint_consts = [NONE; 6];
    for (h, id) in hint_consts.iter_mut().enumerate() {
        if hint_values[h] != NONE {
            *id = match m.lookup_constant(Ty::I32, Const::Int(h as u64 + 1)) {
                Some(c) => const_ids[(c.0 & 0x3fff_ffff) as usize],
                None => marks.extra_at.get(&(Ty::I32, h as u64 + 1)).map_or(NONE, |&e| extra_ids[e as usize]),
            };
        }
    }
    Ok(Ids { type_ids, fn_ptr_ids, type_list, type_bits, global_ids, func_ids, const_ids, extra_ids, globals, funcs, consts, base: next, md_order, md_ids, precise_node, precise_one, precise_synthetic, hint_string, hint_values, hint_nodes, hint_consts, loop_string, loops })
}

/// Give every live instruction of a plan its value id, in block order.
#[inline(never)]
fn number_body(plan: &mut Plan, base: u32, ids: &Ids) {
    let body = plan.body;
    let mut table = vec![NONE; body.insts.len()];
    let mut next = base;
    for (b, data) in body.blocks.iter().enumerate() {
        if plan.block_ids[b] == NONE {
            continue;
        }
        for list in [&data.phis, &data.insts] {
            for &i in list {
                let i = i as usize;
                if plan.live[i] && body.insts[i].ty != Ty::VOID {
                    table[i] = next;
                    next += 1;
                }
            }
        }
    }
    if !plan.fwd.is_empty() {
        for (old, _) in body.replacements() {
            let v = plan.fwd[old as usize];
            let i = (v & 0x3fff_ffff) as usize;
            table[old as usize] = match v >> 30 {
                0 => table[i],
                1 => ids.const_ids.get(i).copied().unwrap_or(NONE),
                2 => ids.global_ids.get(i).copied().unwrap_or(NONE),
                _ => ids.func_ids.get(i).copied().unwrap_or(NONE),
            };
        }
    }
    plan.ids = table;
}

fn string_record(w: &mut Writer, code: u32, s: &str) {
    w.record_head(code, s.len() as u32);
    for c in s.bytes() {
        w.vbr(c as u32, 6);
    }
}

fn write_blockinfo(w: &mut Writer, type_bits: u32) {
    use AbbrevOp::*;
    w.enter_blockinfo();
    w.blockinfo_abbrev(VALUE_SYMTAB_BLOCK, &[Fixed(3), Vbr(8), Array, Fixed(8)]);
    w.blockinfo_abbrev(VALUE_SYMTAB_BLOCK, &[Literal(VST_ENTRY as u64), Vbr(8), Array, Fixed(7)]);
    w.blockinfo_abbrev(VALUE_SYMTAB_BLOCK, &[Literal(VST_ENTRY as u64), Vbr(8), Array, Char6]);
    w.blockinfo_abbrev(VALUE_SYMTAB_BLOCK, &[Literal(VST_BBENTRY as u64), Vbr(8), Array, Char6]);
    w.blockinfo_abbrev(CONSTANTS_BLOCK, &[Literal(CST_SETTYPE as u64), Fixed(type_bits)]);
    w.blockinfo_abbrev(CONSTANTS_BLOCK, &[Literal(CST_INTEGER as u64), Vbr(8)]);
    w.blockinfo_abbrev(CONSTANTS_BLOCK, &[Literal(11), Fixed(4), Fixed(type_bits), Vbr(8)]);
    w.blockinfo_abbrev(CONSTANTS_BLOCK, &[Literal(CST_NULL as u64)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_LOAD as u64), Vbr(6), Fixed(type_bits), Vbr(4), Fixed(1)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_BINOP as u64), Vbr(6), Vbr(6), Fixed(4)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_BINOP as u64), Vbr(6), Vbr(6), Fixed(4), Fixed(7)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_CAST as u64), Vbr(6), Fixed(type_bits), Fixed(4)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_RET as u64)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_RET as u64), Vbr(6)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_UNREACHABLE as u64)]);
    w.blockinfo_abbrev(FUNCTION_BLOCK, &[Literal(FUNC_GEP as u64), Fixed(1), Fixed(type_bits), Array, Vbr(6)]);
    w.exit_block();
}

/// The attribute groups of a function's attribute list: (parameter index, attribute kinds),
/// the function's own attributes under index 0xFFFFFFFF.
fn attr_groups(a: FnAttrs) -> &'static [(u64, &'static [u64])] {
    const FN: u64 = 0xffff_ffff;
    match a {
        FnAttrs::None => &[],
        FnAttrs::Nounwind => &[(FN, &[ATTR_NO_UNWIND])],
        FnAttrs::NounwindReadnone => &[(FN, &[ATTR_NO_UNWIND, ATTR_READ_NONE])],
        FnAttrs::NounwindReadonly => &[(FN, &[ATTR_NO_UNWIND, ATTR_READ_ONLY])],
        FnAttrs::NounwindNoreturn => &[(FN, &[ATTR_NO_RETURN, ATTR_NO_UNWIND])],
        FnAttrs::NounwindNoduplicate => &[(FN, &[ATTR_NO_DUPLICATE, ATTR_NO_UNWIND])],
        FnAttrs::Lifetime => &[(2, &[ATTR_NO_CAPTURE]), (FN, &[ATTR_NO_UNWIND])],
    }
}

/// The attribute groups and lists, in order of first use. Returns the lists: a function's
/// attribute list id is the position of its attributes plus one.
fn write_attributes(w: &mut Writer, m: &Module, ids: &Ids) -> Vec<FnAttrs> {
    let mut lists: Vec<FnAttrs> = Vec::new();
    let mut groups: Vec<(u64, &'static [u64])> = Vec::new();
    for &f in &ids.funcs {
        let a = m.funcs[f as usize].attrs;
        if a != FnAttrs::None && !lists.contains(&a) {
            lists.push(a);
            for g in attr_groups(a) {
                if !groups.contains(g) {
                    groups.push(*g);
                }
            }
        }
    }
    if lists.is_empty() {
        return lists;
    }
    w.enter_block(PARAMATTR_GROUP_BLOCK, 3);
    let mut rec = Vec::new();
    for (g, (index, kinds)) in groups.iter().enumerate() {
        rec.clear();
        rec.extend_from_slice(&[g as u64 + 1, *index]);
        for &k in *kinds {
            rec.extend_from_slice(&[0, k]);
        }
        w.record(PARAMATTR_GRP_ENTRY, &rec);
    }
    w.exit_block();
    w.enter_block(PARAMATTR_BLOCK, 3);
    for a in &lists {
        rec.clear();
        rec.extend(attr_groups(*a).iter().filter_map(|g| groups.iter().position(|x| x == g)).map(|g| g as u64 + 1));
        w.record(PARAMATTR_ENTRY, &rec);
    }
    w.exit_block();
    lists
}

fn write_types(w: &mut Writer, m: &Module, ids: &Ids) -> Result<()> {
    use AbbrevOp::*;
    let bits = ids.type_bits;
    w.enter_block(TYPE_BLOCK_NEW, 4);
    let pointer = [Literal(TYPE_POINTER as u64), Fixed(bits), Literal(0)];
    let function = [Literal(TYPE_FUNCTION as u64), Fixed(1), Array, Fixed(bits)];
    let anon = [Literal(TYPE_STRUCT_ANON as u64), Fixed(1), Array, Fixed(bits)];
    let name = [Literal(TYPE_STRUCT_NAME as u64), Array, Char6];
    let named = [Literal(TYPE_STRUCT_NAMED as u64), Fixed(1), Array, Fixed(bits)];
    let array = [Literal(TYPE_ARRAY as u64), Vbr(8), Fixed(bits)];
    for a in [&pointer[..], &function, &anon, &name, &named, &array] {
        w.define_abbrev(a);
    }
    w.record(TYPE_NUMENTRY, &[ids.type_list.len() as u64]);
    let id = |t: Ty| ids.type_ids[t.0 as usize] as u64;
    let mut rec: Vec<u64> = Vec::new();
    for e in &ids.type_list {
        rec.clear();
        let t = match *e {
            TypeEntry::FnPtr(f) => {
                w.abbreviated(4, &pointer, &[TYPE_POINTER as u64, id(f), 0]);
                continue;
            }
            TypeEntry::Module(t) => t,
        };
        match &m.types[t.0 as usize] {
            TypeDef::Void => w.record(TYPE_VOID, &[]),
            TypeDef::Half => w.record(TYPE_HALF, &[]),
            TypeDef::Float => w.record(TYPE_FLOAT, &[]),
            TypeDef::Double => w.record(TYPE_DOUBLE, &[]),
            TypeDef::Label => w.record(TYPE_LABEL, &[]),
            TypeDef::Metadata => w.record(TYPE_METADATA, &[]),
            TypeDef::Int(b) => w.record(TYPE_INTEGER, &[*b as u64]),
            TypeDef::Ptr { to, space: 0 } => w.abbreviated(4, &pointer, &[TYPE_POINTER as u64, id(*to), 0]),
            TypeDef::Ptr { to, space } => w.record(TYPE_POINTER, &[id(*to), *space as u64]),
            TypeDef::Array { n, of } => w.abbreviated(9, &array, &[TYPE_ARRAY as u64, *n, id(*of)]),
            TypeDef::Vector { n, of } => w.record(TYPE_VECTOR, &[*n as u64, id(*of)]),
            TypeDef::Func { ret, params } => {
                rec.extend_from_slice(&[TYPE_FUNCTION as u64, 0, id(*ret)]);
                rec.extend(params.iter().map(|&p| id(p)));
                w.abbreviated(5, &function, &rec);
            }
            TypeDef::Struct { name: struct_name, fields } => {
                if let Some(s) = struct_name {
                    if s.is_empty() {
                        return internal("a named struct type with an empty name");
                    }
                    if s.bytes().all(is_char6) {
                        rec.push(TYPE_STRUCT_NAME as u64);
                        rec.extend(s.bytes().map(|c| c as u64));
                        w.abbreviated(7, &name, &rec);
                    } else {
                        string_record(w, TYPE_STRUCT_NAME, s);
                    }
                    rec.clear();
                }
                rec.extend_from_slice(&[if struct_name.is_some() { TYPE_STRUCT_NAMED } else { TYPE_STRUCT_ANON } as u64, 0]);
                rec.extend(fields.iter().map(|&f| id(f)));
                if struct_name.is_some() {
                    w.abbreviated(8, &named, &rec);
                } else {
                    w.abbreviated(6, &anon, &rec);
                }
            }
        }
    }
    w.exit_block();
    Ok(())
}

fn write_module_info(w: &mut Writer, m: &Module, ids: &Ids, groups: &[FnAttrs]) {
    use AbbrevOp::*;
    string_record(w, MODULE_TRIPLE, crate::irprint::TRIPLE);
    string_record(w, MODULE_DATALAYOUT, crate::irprint::DATALAYOUT);
    if !ids.globals.is_empty() {
        let max_align = ids.globals.iter().map(|&g| m.globals[g as usize].align).max().unwrap_or(0);
        let max_type = ids.globals.iter().map(|&g| ids.type_ids[m.globals[g as usize].value_ty.0 as usize]).max().unwrap_or(0);
        let field = |bits: u32| if bits == 0 { Literal(0) } else { Fixed(bits) };
        let align = if max_align == 0 { Literal(0) } else { field(log2_ceil(align_field(max_align) + 1)) };
        let simple = [Literal(MODULE_GLOBALVAR as u64), field(log2_ceil(max_type + 1)), Vbr(6), Vbr(6), Fixed(5), align, Literal(0)];
        w.define_abbrev(&simple);
        for &g in &ids.globals {
            let g = &m.globals[g as usize];
            let init = g.init.map_or(0, |v| ids.const_ids[(v.0 & 0x3fff_ffff) as usize] as u64 + 1);
            let head = [ids.type_ids[g.value_ty.0 as usize] as u64, (g.space as u64) << 2 | 2 | g.constant as u64, init, if g.internal { 3 } else { 0 }, align_field(g.align) as u64, 0];
            if g.unnamed_addr {
                // visibility, thread-local, unnamed_addr, externally initialised, dll storage class, comdat
                let mut rec = head.to_vec();
                rec.extend_from_slice(&[0, 0, 1, 0, 0, 0]);
                w.record(MODULE_GLOBALVAR, &rec);
            } else {
                let mut rec = vec![MODULE_GLOBALVAR as u64];
                rec.extend_from_slice(&head);
                w.abbreviated(4, &simple, &rec);
            }
        }
    }
    for &f in &ids.funcs {
        let f = &m.funcs[f as usize];
        let attrs = groups.iter().position(|&g| g == f.attrs).map_or(0, |g| g as u64 + 1);
        // type, calling convention, is a declaration, linkage, attributes, alignment, section,
        // visibility, gc, unnamed_addr, prologue data, dll storage class, comdat, prefix
        // data, personality
        w.record(MODULE_FUNCTION, &[ids.type_ids[f.ty.0 as usize] as u64, 0, f.body.is_none() as u64, 0, attrs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}

/// The value LLVM's ConstantDataArray holds for an element: the integer zero-extended, the
/// float's bits. None when the element is not a plain number.
fn data_element(m: &Module, v: Value) -> Option<u64> {
    match m.as_const(v)?.c {
        Const::Int(x) | Const::Float(x) => Some(x),
        _ => None,
    }
}

#[inline(never)]
fn write_constants(w: &mut Writer, m: &Module, marks: &Marks, ids: &Ids) -> Result<()> {
    use AbbrevOp::*;
    if ids.consts.is_empty() {
        return Ok(());
    }
    w.enter_block(CONSTANTS_BLOCK, 4);
    let aggregate = [Literal(CST_AGGREGATE as u64), Array, Fixed(log2_ceil(ids.base + 1))];
    let string8 = [Literal(CST_STRING as u64), Array, Fixed(8)];
    let cstring7 = [Literal(CST_CSTRING as u64), Array, Fixed(7)];
    let cstring6 = [Literal(CST_CSTRING as u64), Array, Char6];
    for a in [&aggregate[..], &string8, &cstring7, &cstring6] {
        w.define_abbrev(a);
    }
    let integer = |w: &mut Writer, ty: Ty, v: u64| {
        if v == 0 {
            w.record(CST_NULL, &[]);
        } else {
            w.code(CST_INTEGER_ABBREV);
            w.vbr64(signed_vbr(sext(v, m.bits(ty))), 8);
        }
    };
    // the id and type of a constant's operand
    let operand = |v: Value| -> Option<(u32, Ty)> {
        let i = (v.0 & 0x3fff_ffff) as usize;
        let (id, ty) = match v.0 >> 30 {
            1 => (ids.const_ids.get(i).copied()?, m.consts.get(i)?.ty),
            2 => (ids.global_ids.get(i).copied()?, m.globals.get(i)?.ptr_ty),
            3 if !v.is_none() => (ids.func_ids.get(i).copied()?, Ty(NONE)),
            _ => return None,
        };
        (id != NONE).then_some((id, ty))
    };
    let mut last = Ty(NONE);
    let mut rec: Vec<u64> = Vec::new();
    for e in &ids.consts {
        let (ty, c) = match *e {
            ConstEntry::Extra(i) => {
                let (ty, v) = marks.extra[i as usize];
                (ty, Const::Int(v))
            }
            ConstEntry::Module(i) => (m.consts[i as usize].ty, Const::Undef),
        };
        if ty != last {
            last = ty;
            w.code(CST_SETTYPE_ABBREV);
            w.emit(ids.type_ids[ty.0 as usize], ids.type_bits);
        }
        let c = match *e {
            ConstEntry::Extra(_) => &c,
            ConstEntry::Module(i) => &m.consts[i as usize].c,
        };
        match c {
            Const::Int(v) => integer(w, ty, *v),
            // LLVM writes +0.0 as the null value
            Const::Float(0) | Const::Zero => w.record(CST_NULL, &[]),
            Const::Float(bits) => w.record(CST_FLOAT, &[*bits]),
            Const::Undef => w.record(CST_UNDEF, &[]),
            // a constant expression: a bitcast (see Module::const_bitcast): [opcode, type, value]
            Const::Aggregate(operands) if operands.len() == 1 && matches!(m.types.get(ty.0 as usize), Some(TypeDef::Ptr { .. })) => {
                let from = operand(operands[0]).and_then(|(id, t)| Some((id, ids.type_ids.get(t.0 as usize).copied().filter(|&id| id != NONE)?)));
                let Some((id, t)) = from else { return internal("a bitcast constant of something that is not a global or a constant") };
                w.record(CST_CE_CAST, &[CAST_BITCAST as u64, t as u64, id as u64]);
            }
            // or a getelementptr (see Module::const_gep):
            // [source element type, (type, value) of the base and of every index]
            Const::Aggregate(operands) if matches!(m.types.get(ty.0 as usize), Some(TypeDef::Ptr { .. })) => {
                rec.clear();
                let type_id = |t: Ty| ids.type_ids.get(t.0 as usize).copied().filter(|&id| id != NONE);
                let source = operands.first().and_then(|&base| operand(base)).and_then(|(_, t)| match m.types.get(t.0 as usize) {
                    Some(TypeDef::Ptr { to, .. }) => type_id(*to),
                    _ => None,
                });
                let Some(source) = source else { return internal("a getelementptr constant whose base is not a global or a pointer constant") };
                rec.push(source as u64);
                for &v in operands {
                    match operand(v).and_then(|(id, t)| Some((id, type_id(t)?))) {
                        Some((id, t)) => rec.extend_from_slice(&[t as u64, id as u64]),
                        None => return internal("a getelementptr constant with an operand that is not a constant"),
                    }
                }
                w.record(CST_CE_INBOUNDS_GEP, &rec);
            }
            Const::Aggregate(fields) => {
                rec.clear();
                // an array of plain numbers is a ConstantDataArray in LLVM: i8 arrays as
                // strings, the others as CST_CODE_DATA
                let element = match m.types.get(ty.0 as usize) {
                    Some(TypeDef::Array { of, .. }) if matches!(*of, Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::F32 | Ty::F64) => Some(*of),
                    _ => None,
                };
                let plain = element.is_some() && !fields.is_empty() && fields.iter().all(|&f| data_element(m, f).is_some());
                if plain {
                    rec.extend(fields.iter().filter_map(|&f| data_element(m, f)));
                    if rec.iter().all(|&x| x == 0) {
                        w.record(CST_NULL, &[]);
                    } else if element == Some(Ty::I8) {
                        let cstring = rec.last() == Some(&0) && rec[..rec.len() - 1].iter().all(|&x| x != 0);
                        if !cstring {
                            rec.insert(0, CST_STRING as u64);
                            w.abbreviated(9, &string8, &rec);
                        } else {
                            rec.pop();
                            let (id, abbrev) = if rec.iter().all(|&x| is_char6(x as u8)) { (11, &cstring6) } else if rec.iter().all(|&x| x < 128) { (10, &cstring7) } else { (0, &cstring7) };
                            if id == 0 {
                                w.record(CST_CSTRING, &rec);
                            } else {
                                rec.insert(0, CST_CSTRING as u64);
                                w.abbreviated(id, abbrev, &rec);
                            }
                        }
                    } else {
                        w.record(CST_DATA, &rec);
                    }
                } else {
                    rec.push(CST_AGGREGATE as u64);
                    for &f in fields {
                        match operand(f) {
                            Some((id, _)) => rec.push(id as u64),
                            None => return internal("an aggregate constant holds a value that is not a constant"),
                        }
                    }
                    w.abbreviated(8, &aggregate, &rec);
                }
            }
        }
    }
    w.exit_block();
    Ok(())
}

#[inline(never)]
fn write_metadata(w: &mut Writer, m: &Module, ids: &Ids) -> Result<()> {
    use AbbrevOp::*;
    if !(ids.md_order.is_empty() && m.named_md.is_empty() && !ids.precise_synthetic && ids.hint_string == NONE && ids.loop_string == NONE) {
        w.enter_block(METADATA_BLOCK, 3);
        let string = [Literal(METADATA_STRING as u64), Array, Fixed(8)];
        let name = [Literal(METADATA_NAME as u64), Array, Fixed(8)];
        let mut next_abbrev = 4;
        let mut string_id = 0;
        if ids.md_order.iter().any(|&md| matches!(m.md[md as usize], MdNode::Str(_))) {
            w.define_abbrev(&string);
            string_id = next_abbrev;
            next_abbrev += 1;
        }
        if !m.named_md.is_empty() {
            w.define_abbrev(&name);
        }
        let name_id = next_abbrev;
        let mut rec: Vec<u64> = Vec::new();
        for &md in &ids.md_order {
            rec.clear();
            match &m.md[md as usize] {
                MdNode::Str(s) => {
                    rec.push(METADATA_STRING as u64);
                    rec.extend(s.bytes().map(|c| c as u64));
                    w.abbreviated(string_id, &string, &rec);
                }
                MdNode::Val(v) => {
                    let i = (v.0 & 0x3fff_ffff) as usize;
                    let (ty, id) = match v.0 >> 30 {
                        1 => (ids.type_ids[m.consts[i].ty.0 as usize], ids.const_ids[i]),
                        2 => (ids.type_ids[m.globals[i].ptr_ty.0 as usize], ids.global_ids[i]),
                        _ => (ids.fn_ptr_ids.get(&m.funcs[i].ty.0).copied().unwrap_or(NONE), ids.func_ids[i]),
                    };
                    if ty == NONE || id == NONE {
                        return internal("metadata value without a type or an id");
                    }
                    w.record(METADATA_VALUE, &[ty as u64, id as u64]);
                }
                MdNode::Tuple(ops) => {
                    rec.extend(ops.iter().map(|op| op.map_or(0, |o| ids.md_ids[o.0 as usize] as u64 + 1)));
                    w.record(METADATA_NODE, &rec);
                }
            }
        }
        if ids.precise_synthetic {
            w.record(METADATA_VALUE, &[ids.type_ids[Ty::I32.0 as usize] as u64, ids.precise_one as u64]);
            w.record(METADATA_NODE, &[ids.md_order.len() as u64 + 1]);
        }
        if ids.hint_string != NONE {
            // as DXC builds them: distinct, the node itself as its first operand
            rec.clear();
            rec.extend("dx.controlflow.hints".bytes().map(|c| c as u64));
            w.record(METADATA_STRING, &rec);
            for h in 0..6 {
                if ids.hint_values[h] != NONE {
                    w.record(METADATA_VALUE, &[ids.type_ids[Ty::I32.0 as usize] as u64, ids.hint_consts[h] as u64]);
                }
            }
            for set in 1..64 {
                if ids.hint_nodes[set] != NONE {
                    rec.clear();
                    rec.extend_from_slice(&[ids.hint_nodes[set] as u64 + 1, ids.hint_string as u64 + 1]);
                    rec.extend((0..6).filter(|h| set >> h & 1 != 0).map(|h| ids.hint_values[h] as u64 + 1));
                    w.record(METADATA_DISTINCT_NODE, &rec);
                }
            }
        }
        if ids.loop_string != NONE {
            rec.clear();
            rec.extend("llvm.loop.unroll.disable".bytes().map(|c| c as u64));
            w.record(METADATA_STRING, &rec);
            w.record(METADATA_NODE, &[ids.loop_string as u64 + 1]);
            for k in 0..ids.loops as u64 {
                let me = ids.loop_string as u64 + 2 + k;
                w.record(METADATA_DISTINCT_NODE, &[me + 1, ids.loop_string as u64 + 2]);
            }
        }
        for (named, nodes) in &m.named_md {
            rec.clear();
            rec.push(METADATA_NAME as u64);
            rec.extend(named.bytes().map(|c| c as u64));
            w.abbreviated(name_id, &name, &rec);
            rec.clear();
            for n in nodes {
                if !matches!(m.md.get(n.0 as usize), Some(MdNode::Tuple(_))) {
                    return internal(format!("named metadata {named} has an operand that is not a node"));
                }
                rec.push(ids.md_ids[n.0 as usize] as u64);
            }
            w.record(METADATA_NAMED_NODE, &rec);
        }
        w.exit_block();
    }
    w.enter_block(METADATA_BLOCK, 3);
    for (k, kind) in MD_KINDS.iter().enumerate() {
        w.record_head(METADATA_KIND, kind.len() as u32 + 1);
        w.vbr(k as u32, 6);
        for c in kind.bytes() {
            w.vbr(c as u32, 6);
        }
    }
    w.exit_block();
    Ok(())
}

fn write_symbols(w: &mut Writer, m: &Module, ids: &Ids) {
    use AbbrevOp::*;
    let mut names: Vec<(&str, u32)> = Vec::with_capacity(ids.globals.len() + ids.funcs.len());
    names.extend(ids.globals.iter().map(|&g| (m.globals[g as usize].name.as_str(), ids.global_ids[g as usize])));
    names.extend(ids.funcs.iter().map(|&f| (m.funcs[f as usize].name.as_str(), ids.func_ids[f as usize])));
    if names.is_empty() {
        return;
    }
    // DXC sorts the table by name so that its output does not depend on hash order
    names.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    w.enter_block(VALUE_SYMTAB_BLOCK, 4);
    let mut rec: Vec<u64> = Vec::new();
    for (name, id) in names {
        rec.clear();
        rec.extend_from_slice(&[VST_ENTRY as u64, id as u64]);
        rec.extend(name.bytes().map(|c| c as u64));
        if name.bytes().all(is_char6) {
            w.abbreviated(VST_ENTRY_6, &[Literal(VST_ENTRY as u64), Vbr(8), Array, Char6], &rec);
        } else if name.bytes().all(|c| c < 128) {
            w.abbreviated(VST_ENTRY_7, &[Literal(VST_ENTRY as u64), Vbr(8), Array, Fixed(7)], &rec);
        } else {
            w.abbreviated(VST_ENTRY_8, &[Fixed(3), Vbr(8), Array, Fixed(8)], &rec);
        }
    }
    w.exit_block();
}

fn bin_code(op: BinOp) -> u32 {
    match op {
        BinOp::Add | BinOp::FAdd => 0,
        BinOp::Sub | BinOp::FSub => 1,
        BinOp::Mul | BinOp::FMul => 2,
        BinOp::UDiv => 3,
        BinOp::SDiv | BinOp::FDiv => 4,
        BinOp::URem => 5,
        BinOp::SRem | BinOp::FRem => 6,
        BinOp::Shl => 7,
        BinOp::LShr => 8,
        BinOp::AShr => 9,
        BinOp::And => 10,
        BinOp::Or => 11,
        BinOp::Xor => 12,
    }
}

/// LLVM's optimisation flags field of a binary operator: nuw 1 / nsw 2 on add, sub, mul and
/// shl; exact 1 on the divisions and right shifts; all five fast-math bits for `fast`.
fn bin_flags(op: BinOp, flags: u8) -> u32 {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => (flags & flag::NUW != 0) as u32 | ((flags & flag::NSW != 0) as u32) << 1,
        BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr => (flags & flag::EXACT != 0) as u32,
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem => {
            if flags & flag::FAST != 0 {
                31
            } else {
                0
            }
        }
        _ => 0,
    }
}

fn cast_code(op: CastOp) -> u32 {
    match op {
        CastOp::Trunc => 0,
        CastOp::ZExt => 1,
        CastOp::SExt => 2,
        CastOp::FPToUI => 3,
        CastOp::FPToSI => 4,
        CastOp::UIToFP => 5,
        CastOp::SIToFP => 6,
        CastOp::FPTrunc => 7,
        CastOp::FPExt => 8,
        CastOp::BitCast => 11,
    }
}

struct FnWriter<'a, 'b> {
    m: &'a Module,
    ids: &'b Ids,
    marks: &'b Marks<'a>,
    plan: &'b Plan<'a>,
    /// the value id of the function's first instruction
    base: u32,
}

impl<'a, 'b> FnWriter<'a, 'b> {
    /// The alignment of an alloca, load or store of a value of type `ty`.
    #[inline(always)]
    fn align(&self, i: u32, ty: Ty) -> u32 {
        let aligns = &self.plan.body.aligns;
        let known = if aligns.is_empty() { None } else { aligns.get(&i).copied() };
        align_field(known.unwrap_or_else(|| align_of(self.m, ty)))
    }

    fn name(&self) -> &str {
        &self.m.funcs[self.plan.func as usize].name
    }

    #[cold]
    fn unwritten<T>(&self, at: u32) -> Result<T> {
        internal(format!("{}: instruction {at} uses a value that is not written (dead, in an unreachable block, or never placed in a block)", self.name()))
    }

    /// The value id of an operand, NONE when nothing written stands for it.
    #[inline(always)]
    fn id(&self, v: Value) -> u32 {
        let i = (v.0 & 0x3fff_ffff) as usize;
        let table = match v.0 >> 30 {
            0 => &self.plan.ids,
            1 => &self.ids.const_ids,
            2 => &self.ids.global_ids,
            _ => &self.ids.func_ids,
        };
        table.get(i).copied().unwrap_or(NONE)
    }

    /// The type id of an instruction operand (used for forward references, which only
    /// instructions can be).
    #[inline(always)]
    fn fwd_ty(&self, v: Value) -> u32 {
        let t = self.plan.body.insts[v.0 as usize].ty;
        self.ids.type_ids[t.0 as usize]
    }

    /// LLVM's PushValueAndType: the relative id, then the type when it is a forward
    /// reference.
    #[inline(always)]
    fn typed(&self, w: &mut Writer, cur: u32, v: Value, id: u32) {
        w.vbr(cur.wrapping_sub(id), 6);
        if id >= cur {
            w.vbr(self.fwd_ty(v), 6);
        }
    }

    fn block(&self, b: u32) -> Result<u32> {
        match self.plan.block_ids.get(b as usize) {
            Some(&id) if id != NONE => Ok(id),
            _ => internal(format!("{}: branch to a block that does not exist", self.name())),
        }
    }

    fn pointee(&self, v: Value) -> Result<Ty> {
        let v = if v.is_inst() && !self.plan.fwd.is_empty() { Value(self.plan.fwd[v.0 as usize]) } else { v };
        match value_ty(self.m, self.plan.body, v).and_then(|t| self.m.types.get(t.0 as usize)) {
            Some(TypeDef::Ptr { to, .. }) => Ok(*to),
            _ => internal(format!("{}: address operand that is not a pointer", self.name())),
        }
    }

    /// `attachments` gets (instruction index, metadata kind, node) of what has metadata.
    /// `loops` counts the `!llvm.loop` nodes used so far.
    fn write(&self, out: &mut Writer, attachments: &mut Vec<(u32, u32, u32)>, loops: &mut u32) -> Result<()> {
        // the bit writer is a local of the loop while it runs, so that its accumulator lives
        // in registers and not behind a reference
        let mut w = std::mem::replace(out, Writer::new(0));
        let done = self.instructions(&mut w, attachments, loops);
        *out = w;
        done
    }

    #[inline(always)]
    fn instructions(&self, w: &mut Writer, attachments: &mut Vec<(u32, u32, u32)>, loops: &mut u32) -> Result<()> {
        let body = self.plan.body;
        // the value id the next value-producing instruction gets
        let mut next = self.base;
        let ids = self.ids;
        let bits = ids.type_bits;
        // index among the written instructions, which is what attachments name
        let mut index = 0u32;
        for (b, data) in body.blocks.iter().enumerate() {
            if self.plan.block_ids[b] == NONE {
                continue;
            }
            let last = data.insts.last().copied();
            for list in [&data.phis, &data.insts] {
                for &i in list {
                    if !self.plan.live[i as usize] {
                        continue;
                    }
                    let inst = &body.insts[i as usize];
                    let cur = next;
                    if inst.op.is_terminator() && Some(i) != last {
                        return internal(format!("{}: block {b} has a terminator before its end", self.name()));
                    }
                    match inst.op {
                        Op::Bin(op) => {
                            let (a, bid) = (self.id(inst.a), self.id(inst.b));
                            if a == NONE || bid == NONE {
                                return self.unwritten(i);
                            }
                            let flags = bin_flags(op, inst.flags);
                            if a < cur {
                                w.code(if flags == 0 { FN_BINOP_ABBREV } else { FN_BINOP_FLAGS_ABBREV });
                                w.vbr(cur - a, 6);
                                w.vbr(cur.wrapping_sub(bid), 6);
                                w.emit(bin_code(op), 4);
                                if flags != 0 {
                                    w.emit(flags, 7);
                                }
                            } else {
                                w.record_head(FUNC_BINOP, 4 + (flags != 0) as u32);
                                self.typed(w, cur, inst.a, a);
                                w.vbr(cur.wrapping_sub(bid), 6);
                                w.vbr(bin_code(op), 6);
                                if flags != 0 {
                                    w.vbr(flags, 6);
                                }
                            }
                        }
                        Op::Cast(op) => {
                            let a = self.id(inst.a);
                            if a == NONE {
                                return self.unwritten(i);
                            }
                            let ty = ids.type_ids[inst.ty.0 as usize];
                            if a < cur {
                                w.code(FN_CAST_ABBREV);
                                w.vbr(cur - a, 6);
                                w.emit(ty, bits);
                                w.emit(cast_code(op), 4);
                            } else {
                                w.record_head(FUNC_CAST, 4);
                                self.typed(w, cur, inst.a, a);
                                w.vbr(ty, 6);
                                w.vbr(cast_code(op), 6);
                            }
                        }
                        Op::ICmp(_) | Op::FCmp(_) => {
                            let (a, bid) = (self.id(inst.a), self.id(inst.b));
                            if a == NONE || bid == NONE {
                                return self.unwritten(i);
                            }
                            let (pred, fast) = match inst.op {
                                Op::ICmp(p) => (p as u32, false),
                                Op::FCmp(p) => (p as u32, inst.flags & flag::FAST != 0),
                                _ => (0, false),
                            };
                            w.record_head(FUNC_CMP2, 3 + (a >= cur) as u32 + fast as u32);
                            self.typed(w, cur, inst.a, a);
                            w.vbr(cur.wrapping_sub(bid), 6);
                            w.vbr(pred, 6);
                            if fast {
                                w.vbr(31, 6);
                            }
                        }
                        Op::Select => {
                            let (c, t, f) = (self.id(inst.a), self.id(inst.b), self.id(inst.c));
                            if c == NONE || t == NONE || f == NONE {
                                return self.unwritten(i);
                            }
                            w.record_head(FUNC_VSELECT, 3 + (t >= cur) as u32 + (c >= cur) as u32);
                            self.typed(w, cur, inst.b, t);
                            w.vbr(cur.wrapping_sub(f), 6);
                            self.typed(w, cur, inst.a, c);
                        }
                        Op::ExtractValue => {
                            let a = self.id(inst.a);
                            if a == NONE {
                                return self.unwritten(i);
                            }
                            w.record_head(FUNC_EXTRACTVAL, 2 + (a >= cur) as u32);
                            self.typed(w, cur, inst.a, a);
                            w.vbr(inst.b.0, 6);
                        }
                        Op::Call => {
                            let callee = inst.a;
                            let (first, n) = (inst.b.0 as usize, inst.c.0 as usize);
                            let (Some(f), Some(args)) = (self.m.funcs.get((callee.0 & 0x3fff_ffff) as usize).filter(|_| callee.0 >> 30 == 3), body.operands.get(first..first.saturating_add(n))) else {
                                return internal(format!("{}: call of something that is not a function", self.name()));
                            };
                            let fid = ids.func_ids[(callee.0 & 0x3fff_ffff) as usize];
                            match &self.m.types[f.ty.0 as usize] {
                                TypeDef::Func { params, .. } if params.len() == n => {}
                                _ => return internal(format!("{}: call of {} with {n} arguments", self.name(), f.name)),
                            }
                            // attributes, calling convention with the explicit-type bit, function
                            // type, callee, arguments
                            w.record_head(FUNC_CALL, 4 + n as u32);
                            w.vbr(0, 6);
                            w.vbr(1 << 15, 6);
                            w.vbr(ids.type_ids[f.ty.0 as usize], 6);
                            w.vbr(cur - fid, 6);
                            for &a in args {
                                let id = self.id(a);
                                if id == NONE {
                                    return self.unwritten(i);
                                }
                                w.vbr(cur.wrapping_sub(id), 6);
                            }
                            if inst.flags & flag::PRECISE != 0 {
                                attachments.push((index, MD_KIND_PRECISE as u32, ids.precise_node));
                            }
                        }
                        Op::Load => {
                            let a = self.id(inst.a);
                            if a == NONE {
                                return self.unwritten(i);
                            }
                            let ty = ids.type_ids[inst.ty.0 as usize];
                            let align = self.align(i, inst.ty);
                            if a < cur {
                                w.code(FN_LOAD_ABBREV);
                                w.vbr(cur - a, 6);
                                w.emit(ty, bits);
                                w.vbr(align, 4);
                                w.emit(0, 1);
                            } else {
                                w.record_head(FUNC_LOAD, 5);
                                self.typed(w, cur, inst.a, a);
                                w.vbr(ty, 6);
                                w.vbr(align, 6);
                                w.vbr(0, 6);
                            }
                        }
                        Op::Store => {
                            let (p, v) = (self.id(inst.a), self.id(inst.b));
                            if p == NONE || v == NONE {
                                return self.unwritten(i);
                            }
                            let align = self.align(i, self.pointee(inst.a)?);
                            w.record_head(FUNC_STORE, 4 + (p >= cur) as u32 + (v >= cur) as u32);
                            self.typed(w, cur, inst.a, p);
                            self.typed(w, cur, inst.b, v);
                            w.vbr(align, 6);
                            w.vbr(0, 6);
                        }
                        Op::Gep => {
                            let (first, n) = (inst.b.0 as usize, inst.c.0 as usize);
                            let Some(indices) = body.operands.get(first..first.saturating_add(n)) else { return internal(format!("{}: getelementptr without its indices", self.name())) };
                            let base = self.id(inst.a);
                            if base == NONE {
                                return self.unwritten(i);
                            }
                            let source = ids.type_ids[self.pointee(inst.a)?.0 as usize];
                            let mut count = 1 + n as u32 + (base >= cur) as u32;
                            for &x in indices {
                                let id = self.id(x);
                                if id == NONE {
                                    return self.unwritten(i);
                                }
                                count += (id >= cur) as u32;
                            }
                            w.code(FN_GEP_ABBREV);
                            w.emit((inst.flags & flag::NO_INBOUNDS == 0) as u32, 1);
                            w.emit(source, bits);
                            w.vbr(count, 6);
                            self.typed(w, cur, inst.a, base);
                            for &x in indices {
                                let id = self.id(x);
                                self.typed(w, cur, x, id);
                            }
                        }
                        Op::Alloca => {
                            let Some(TypeDef::Ptr { to, .. }) = self.m.types.get(inst.ty.0 as usize) else { return internal(format!("{}: alloca whose type is not a pointer", self.name())) };
                            let one = match self.m.lookup_constant(Ty::I32, Const::Int(1)) {
                                Some(c) => self.id(c),
                                None => self.marks.extra_at.get(&(Ty::I32, 1)).map_or(NONE, |&e| ids.extra_ids[e as usize]),
                            };
                            // allocated type, type and id of the count, alignment with the
                            // explicit-type bit
                            w.record_head(FUNC_ALLOCA, 4);
                            w.vbr(ids.type_ids[to.0 as usize], 6);
                            w.vbr(ids.type_ids[Ty::I32.0 as usize], 6);
                            w.vbr(one, 6);
                            w.vbr(self.align(i, *to) | 1 << 6, 6);
                            if inst.flags & flag::PRECISE != 0 {
                                attachments.push((index, MD_KIND_PRECISE as u32, ids.precise_node));
                            }
                        }
                        Op::Phi => {
                            let Some(incoming) = body.phis.get(inst.a.0 as usize) else { return internal(format!("{}: phi without its list", self.name())) };
                            let n = incoming.iter().filter(|(_, from)| self.plan.block_ids.get(from.0 as usize).is_some_and(|&b| b != NONE)).count();
                            w.record_head(FUNC_PHI, 1 + 2 * n as u32);
                            w.vbr(ids.type_ids[inst.ty.0 as usize], 6);
                            for &(v, from) in incoming {
                                let Some(&from) = self.plan.block_ids.get(from.0 as usize).filter(|&&b| b != NONE) else { continue };
                                let id = self.id(v);
                                if id == NONE {
                                    return self.unwritten(i);
                                }
                                w.vbr64(signed_vbr(cur as i32 as i64 - id as i32 as i64), 6);
                                w.vbr(from, 6);
                            }
                        }
                        Op::AtomicRmw(op) => {
                            let (p, v) = (self.id(inst.a), self.id(inst.b));
                            if p == NONE || v == NONE {
                                return self.unwritten(i);
                            }
                            // pointer, value, operation, volatile, ordering (seq_cst), scope
                            // (cross-thread)
                            w.record_head(FUNC_ATOMICRMW, 6 + (p >= cur) as u32);
                            self.typed(w, cur, inst.a, p);
                            w.vbr(cur.wrapping_sub(v), 6);
                            w.vbr(op as u32, 6);
                            w.vbr(0, 6);
                            w.vbr(6, 6);
                            w.vbr(1, 6);
                        }
                        Op::CmpXchg => {
                            let (p, e, n) = (self.id(inst.a), self.id(inst.b), self.id(inst.c));
                            if p == NONE || e == NONE || n == NONE {
                                return self.unwritten(i);
                            }
                            // pointer, expected, new, volatile, success ordering, scope, failure
                            // ordering, weak
                            w.record_head(FUNC_CMPXCHG, 8 + (p >= cur) as u32 + (e >= cur) as u32);
                            self.typed(w, cur, inst.a, p);
                            self.typed(w, cur, inst.b, e);
                            w.vbr(cur.wrapping_sub(n), 6);
                            for x in [0, 6, 1, 6, 0] {
                                w.vbr(x, 6);
                            }
                        }
                        Op::Br => {
                            let to = self.block(inst.a.0)?;
                            w.record_head(FUNC_BR, 1);
                            w.vbr(to, 6);
                            if inst.flags & flag::LOOP_UNROLL_DISABLE != 0 {
                                attachments.push((index, MD_KIND_LOOP as u32, ids.loop_string + 2 + *loops));
                                *loops += 1;
                            }
                        }
                        Op::CondBr => {
                            let c = self.id(inst.a);
                            if c == NONE {
                                return self.unwritten(i);
                            }
                            let (t, f) = (self.block(inst.b.0)?, self.block(inst.c.0)?);
                            w.record_head(FUNC_BR, 3);
                            w.vbr(t, 6);
                            w.vbr(f, 6);
                            w.vbr(cur.wrapping_sub(c), 6);
                            if inst.flags & flag::HINTS != 0 {
                                attachments.push((index, MD_KIND_HINTS as u32, ids.hint_nodes[(inst.flags & flag::HINTS) as usize]));
                            }
                            if inst.flags & flag::LOOP_UNROLL_DISABLE != 0 {
                                attachments.push((index, MD_KIND_LOOP as u32, ids.loop_string + 2 + *loops));
                                *loops += 1;
                            }
                        }
                        Op::Switch => {
                            let c = self.id(inst.a);
                            if c == NONE {
                                return self.unwritten(i);
                            }
                            let cond = if inst.a.is_inst() && !self.plan.fwd.is_empty() { Value(self.plan.fwd[inst.a.0 as usize]) } else { inst.a };
                            let (Some(ty), Some(cases)) = (value_ty(self.m, body, cond), body.switches.get(inst.c.0 as usize)) else { return internal(format!("{}: malformed switch", self.name())) };
                            let default = self.block(inst.b.0)?;
                            if inst.flags & flag::HINTS != 0 {
                                attachments.push((index, MD_KIND_HINTS as u32, ids.hint_nodes[(inst.flags & flag::HINTS) as usize]));
                            }
                            // condition type, condition, default, then (case constant id, block)
                            w.record_head(FUNC_SWITCH, 3 + 2 * cases.len() as u32);
                            w.vbr(ids.type_ids[ty.0 as usize], 6);
                            w.vbr(cur.wrapping_sub(c), 6);
                            w.vbr(default, 6);
                            for &(v, to) in cases {
                                let id = match self.m.lookup_constant(ty, Const::Int(v)) {
                                    Some(k) => self.id(k),
                                    None => self.marks.extra_at.get(&(ty, v)).map_or(NONE, |&e| ids.extra_ids[e as usize]),
                                };
                                if id == NONE {
                                    return internal(format!("{}: switch case without its constant", self.name()));
                                }
                                let to = self.block(to.0)?;
                                w.vbr(id, 6);
                                w.vbr(to, 6);
                            }
                        }
                        Op::Ret => w.code(FN_RET_VOID_ABBREV),
                        Op::Unreachable => w.code(FN_UNREACHABLE_ABBREV),
                    }
                    if inst.ty != Ty::VOID {
                        next += 1;
                    }
                    index += 1;
                }
            }
        }
        Ok(())
    }
}

/// The bitcode of the module (with the `BC 0xC0DE` magic, without the DXIL program header).
pub fn write(m: &Module) -> Result<(Vec<u8>, Counts)> {
    write_live(m, &module_liveness(m)?, Vec::new())
}

/// [`write`] for a sealed module: its liveness is computed already.
pub fn write_sealed(s: &Sealed) -> Result<(Vec<u8>, Counts)> {
    write_live(s.module(), s.lives(), Vec::new())
}

/// [`write_sealed`] with the bitcode appended to `out`: to what [`crate::container::open`]
/// starts, so that the bitcode is written where the container has it.
pub fn write_sealed_into(s: &Sealed, out: Vec<u8>) -> Result<(Vec<u8>, Counts)> {
    write_live(s.module(), s.lives(), out)
}

/// `lives`: the liveness of each defined function of `m`, by function index. The bitcode
/// goes to the end of `out`.
fn write_live(m: &Module, lives: &[Option<Live>], out: Vec<u8>) -> Result<(Vec<u8>, Counts)> {
    let mut marks = Marks { m, types: vec![false; m.types.len()], consts: vec![false; m.consts.len()], globals: vec![false; m.globals.len()], funcs: vec![false; m.funcs.len()], extra: Vec::new(), extra_at: HashMap::new(), any_precise: false, hints: 0, loops: 0, out_of_range: false };
    let mut plans = Vec::new();
    for (i, f) in m.funcs.iter().enumerate() {
        if let Some(body) = &f.body {
            marks.funcs[i] = true;
            let Some(live) = lives.get(i).and_then(|l| l.as_ref()) else { return internal(format!("{}: no liveness for the body", f.name)) };
            plans.push(analyse(m, i as u32, body, live, &mut marks)?);
        }
    }
    let ids = number(m, &mut marks)?;
    let mut counts = Counts::default();
    for plan in &mut plans {
        number_body(plan, ids.base, &ids);
        counts.instructions += plan.count;
        counts.blocks += plan.blocks as usize;
    }

    let mut w = Writer::appending(out, 4096 + ids.consts.len() * 4 + counts.instructions * 8);
    // 'B' 'C' 0xC0DE
    w.emit(0xdec0_4342, 32);
    w.enter_block(MODULE_BLOCK, 3);
    w.record(MODULE_VERSION, &[1]);
    write_blockinfo(&mut w, ids.type_bits);
    let groups = write_attributes(&mut w, m, &ids);
    write_types(&mut w, m, &ids)?;
    write_module_info(&mut w, m, &ids, &groups);
    write_constants(&mut w, m, &marks, &ids)?;
    write_metadata(&mut w, m, &ids)?;
    write_symbols(&mut w, m, &ids);
    let mut attachments: Vec<(u32, u32, u32)> = Vec::new();
    let mut loops = 0;
    for plan in &plans {
        w.enter_block(FUNCTION_BLOCK, 4);
        w.record(FUNC_DECLAREBLOCKS, &[plan.blocks as u64]);
        attachments.clear();
        FnWriter { m, ids: &ids, marks: &marks, plan, base: ids.base }.write(&mut w, &mut attachments, &mut loops)?;
        if !attachments.is_empty() {
            w.enter_block(METADATA_ATTACHMENT_BLOCK, 3);
            for &(at, kind, node) in &attachments {
                w.record_head(METADATA_ATTACHMENT, 3);
                w.vbr(at, 6);
                w.vbr(kind, 6);
                w.vbr(node, 6);
            }
            w.exit_block();
        }
        w.exit_block();
    }
    w.exit_block();
    Ok((w.finish(), counts))
}
