//! Differential executor: runs the IR `hlsl::lower` builds and the IR DXC builds for the same
//! HLSL on the same inputs, and compares what the two do to the outside world.
//!
//! Both sides are read as LLVM 3.7 text (DXC's `-Fc` disassembly, our `irprint::module`), parsed
//! by the small parser here and interpreted for one thread of a compute shader. Resources
//! are sparse memories whose unread content is a pseudo-random function of (seed, handle,
//! address), so a run reads data everywhere and different seeds take different branches.
//! The *effect trace* of a run is the ordered list of what it does that another thread or
//! the host could see: buffer and texture stores, atomics on resources, stores and atomics
//! on groupshared globals, barriers. Two IRs that compute the same thing give the same
//! trace for every input, however differently they are optimised.
//!
//! Three things make a pass mean more than "16 random runs agreed". The inputs are the
//! seeds plus what a coverage-guided search over DXC's IR adds (`explore`): inputs that
//! take a new branch or turn a comparison for the first time, found by moving the words a
//! run read onto the values it compared them with. Besides the effects, every resource
//! word DXC's IR reads must be read by ours (`unread`), which exposes the addresses a
//! program computes. And `XCL_EXEC_MUTANTS` measures the result: it breaks single
//! instructions of our IR and counts how many of those faults the inputs notice.
//!
//! How to run (`XCL_DXIL_DXC` is the `dxc-linux` folder; without it everything that needs DXC
//! is skipped with a note):
//!
//! ```text
//! cargo test --release --test lower_exec -- --nocapture            # built-in snippets
//! XCL_EXEC_DIR=<dir> ...        every *.hlsl there whose first line says -T cs_6_6, as a table
//! XCL_EXEC_SRC=<file> ...       one file, verbose
//! XCL_EXEC_SRC=<file> XCL_EXEC_REF=<ll> ...   one file against DXC text made earlier
//! ```
//!
//! Knobs: `XCL_EXEC_SEEDS` (16), `XCL_EXEC_SEARCH` (runs of the input search per program; by default 40
//! per block of DXC's IR, between 600 and 50000),
//! `XCL_EXEC_STEPS` (2000000 instructions per run), `XCL_EXEC_LIST=1` (a line per program in the table),
//! `XCL_EXEC_JOBS` (3), `XCL_EXEC_CACHE` (where DXC's text is cached by file name and size),
//! `XCL_EXEC_MUTANTS=<n>` (after a pass, inject n single-instruction faults into our IR and count
//! how many the inputs notice: the measure of how much a pass proves),
//! `XCL_EXEC_TRACE=<n>` (single-file mode: print both traces of input n; `last`: the last effect of
//! every input), `XCL_EXEC_UNREACHED=1` (single-file mode: where the blocks no input reached are),
//! `XCL_EXEC_NANS=1` (on a mismatch, list the float operations that gave NaN in each IR),
//! `XCL_EXEC_NO_READS=1` (compare the effects only, not what is read),
//! `XCL_EXEC_DUMP=<dir>` (write our IR text there).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;

// ------------------------------------------------------------------------------------------
// Hashing: every pseudo-random choice is a pure function of the seed and what is asked.

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn hash(parts: &[u64]) -> u64 {
    let mut h = 0x243F_6A88_85A3_08D3u64;
    for &p in parts {
        h = mix(h ^ p);
    }
    h
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in s.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
    }
    h
}

/// The unread content of a 32-bit word: a mixture, so that tests on the value (zero, sign,
/// single bits, small counts, float classes) go both ways across seeds.
fn content(seed: u64, key: u64) -> u32 {
    let r = hash(&[seed, key]);
    let v = (r >> 32) as u32;
    match r % 16 {
        0..=5 => v,
        6..=8 => v % 17,
        9 => 0,
        10 => u32::MAX,
        11 | 12 => (((v % 65) as f32 - 32.0) * 0.25).to_bits(),
        13 => 1 << (v % 32),
        14 => (1u32 << (v % 32)).wrapping_sub(1),
        _ => v & 0xFF,
    }
}

/// The unread content of a float texel: mostly [0, 1), sometimes any bit pattern.
fn content_f32(seed: u64, key: u64) -> u32 {
    let r = hash(&[seed, key, 0xF]);
    match r % 8 {
        0 => (r >> 32) as u32,
        1 => (((r >> 32) % 65) as f32 - 32.0).to_bits(),
        _ => (((r >> 40) as f32) / 16_777_216.0).to_bits(),
    }
}

// ------------------------------------------------------------------------------------------
// Types of the IR subset. Memory is counted in cells: one scalar is one cell.

type TyId = u32;
const T_VOID: TyId = 0;
const T_I1: TyId = 1;
const T_I8: TyId = 2;
const T_I16: TyId = 3;
const T_I32: TyId = 4;
const T_I64: TyId = 5;
const T_HALF: TyId = 6;
const T_FLOAT: TyId = 7;
const T_DOUBLE: TyId = 8;
const T_HANDLE: TyId = 9;
const T_LABEL: TyId = 10;
const T_METADATA: TyId = 11;

#[derive(Clone, Debug, PartialEq)]
enum TyDef {
    Void,
    Int(u8),
    Half,
    Float,
    Double,
    /// `%dx.types.Handle`: a scalar here (the identity of the resource).
    Handle,
    Label,
    Metadata,
    Ptr(TyId),
    Array(u32, TyId),
    Struct(Vec<TyId>),
    Opaque,
}

struct Types {
    defs: Vec<TyDef>,
    ptr_of: Vec<TyId>,
    arrays: HashMap<(u32, TyId), TyId>,
    named: HashMap<String, TyId>,
}

impl Types {
    fn new() -> Self {
        let mut t = Types { defs: Vec::new(), ptr_of: Vec::new(), arrays: HashMap::new(), named: HashMap::new() };
        for d in [TyDef::Void, TyDef::Int(1), TyDef::Int(8), TyDef::Int(16), TyDef::Int(32), TyDef::Int(64), TyDef::Half, TyDef::Float, TyDef::Double, TyDef::Handle, TyDef::Label, TyDef::Metadata] {
            t.add(d);
        }
        t
    }

    fn add(&mut self, d: TyDef) -> TyId {
        self.defs.push(d);
        self.ptr_of.push(0);
        (self.defs.len() - 1) as TyId
    }

    fn ptr(&mut self, to: TyId) -> TyId {
        if self.ptr_of[to as usize] == 0 {
            let p = self.add(TyDef::Ptr(to));
            self.ptr_of[to as usize] = p;
        }
        self.ptr_of[to as usize]
    }

    fn array(&mut self, n: u32, of: TyId) -> TyId {
        if let Some(&t) = self.arrays.get(&(n, of)) {
            return t;
        }
        let t = self.add(TyDef::Array(n, of));
        self.arrays.insert((n, of), t);
        t
    }

    fn named(&mut self, name: &str) -> TyId {
        if name == "dx.types.Handle" {
            return T_HANDLE;
        }
        if let Some(&t) = self.named.get(name) {
            return t;
        }
        let t = self.add(TyDef::Opaque);
        self.named.insert(name.to_string(), t);
        t
    }

    fn cells(&self, t: TyId) -> u64 {
        match &self.defs[t as usize] {
            TyDef::Void | TyDef::Label | TyDef::Metadata | TyDef::Opaque => 0,
            TyDef::Array(n, of) => (*n as u64).saturating_mul(self.cells(*of)),
            TyDef::Struct(fields) => fields.iter().map(|f| self.cells(*f)).sum(),
            _ => 1,
        }
    }

    fn is_agg(&self, t: TyId) -> bool {
        matches!(self.defs[t as usize], TyDef::Struct(_) | TyDef::Array(..))
    }

    /// 1..64 for integers, 64 for pointers and handles (compared and copied whole).
    fn int_bits(&self, t: TyId) -> R<u8> {
        match self.defs[t as usize] {
            TyDef::Int(b) => Ok(b),
            TyDef::Ptr(_) | TyDef::Handle => Ok(64),
            _ => Err(format!("an integer type was expected, found {:?}", self.defs[t as usize])),
        }
    }

    fn fp_bits(&self, t: TyId) -> R<u8> {
        match self.defs[t as usize] {
            TyDef::Half => Ok(16),
            TyDef::Float => Ok(32),
            TyDef::Double => Ok(64),
            _ => Err(format!("a floating-point type was expected, found {:?}", self.defs[t as usize])),
        }
    }
}

// ------------------------------------------------------------------------------------------
// The parsed program.

#[derive(Clone, Copy, Debug, PartialEq)]
enum Opd {
    Reg(u32),
    /// A constant: the bits of a scalar, a pointer, or the pool index of an aggregate.
    Imm(u64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    UDiv,
    SDiv,
    URem,
    SRem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum IPred {
    Eq,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FPred {
    False,
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
    Uno,
    True,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CastOp {
    Trunc,
    SExt,
    FpToUi,
    FpToSi,
    UiToFp,
    SiToFp,
    FpConv,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum RmwOp {
    Xchg,
    Add,
    Sub,
    And,
    Nand,
    Or,
    Xor,
    Max,
    Min,
    UMax,
    UMin,
}

const NONE: u32 = u32::MAX;

/// Overload of a dx.op, from the suffix of its name: integer widths as they are, floats
/// with 100 added.
const OV_F16: u8 = 116;
const OV_F32: u8 = 132;
const OV_F64: u8 = 164;

#[derive(Clone, Copy, Debug)]
enum Inst {
    Bin { op: BinOp, w: u8, dst: u32, a: Opd, b: Opd },
    FBin { op: FOp, w: u8, dst: u32, a: Opd, b: Opd },
    ICmp { pred: IPred, w: u8, dst: u32, a: Opd, b: Opd },
    FCmp { pred: FPred, w: u8, dst: u32, a: Opd, b: Opd },
    Cast { op: CastOp, from: u8, to: u8, dst: u32, a: Opd },
    Copy { dst: u32, a: Opd },
    Select { dst: u32, c: Opd, a: Opd, b: Opd },
    /// `first..first + n` of `Module::phi_in`.
    Phi { dst: u32, first: u32, n: u32 },
    Br { to: u32 },
    CondBr { c: Opd, t: u32, f: u32 },
    /// `first..first + n` of `Module::cases`.
    Switch { v: Opd, default: u32, first: u32, n: u32 },
    Ret,
    Unreachable,
    /// A dx.op: `args` is `first..first + n` of `Module::args` (the opcode included), `agg` the
    /// pool slot of a struct result, `name` indexes `Module::names`.
    DxOp { op: u16, ovl: u8, dst: u32, first: u32, n: u32, agg: u32, name: u32 },
    /// A call the interpreter has no model for: reaching it fails the run.
    Unknown { name: u32 },
    ExtractValue { dst: u32, a: Opd, idx: u8 },
    Load { dst: u32, p: Opd },
    Store { v: Opd, p: Opd },
    /// `base + off + sum(index * stride)` in cells; the dynamic indices are `first..first + n`
    /// of `Module::gep`.
    Gep { dst: u32, base: Opd, off: i32, first: u32, n: u32 },
    Alloca { dst: u32, ptr: u64 },
    AtomicRmw { op: RmwOp, w: u8, dst: u32, p: Opd, v: Opd },
    CmpXchg { dst: u32, agg: u32, p: Opd, cmp: Opd, new: Opd },
}

/// A global or an alloca: `len` cells at `start`. A pointer is `region << 32 | offset`, so an
/// access out of bounds is the same event whatever the layout (loads give 0, stores vanish).
#[derive(Clone, Copy, Debug)]
struct Region {
    start: u32,
    len: u32,
    /// index in `Module::shared` for a groupshared global
    shared: u32,
}

struct Module {
    insts: Vec<Inst>,
    /// first instruction of each block
    blocks: Vec<u32>,
    phi_in: Vec<(Opd, u32)>,
    cases: Vec<(u64, u32)>,
    args: Vec<Opd>,
    gep: Vec<(Opd, i32, u8)>,
    nregs: u32,
    /// constant aggregates, then one slot per instruction with a struct result
    aggs: Vec<[u64; 5]>,
    regions: Vec<Region>,
    mem: Vec<u64>,
    names: Vec<String>,
    /// groupshared globals: the name as written in HLSL
    shared: Vec<String>,
    /// the constants the program compares against: what the input search tries first
    dict: Vec<u32>,
    /// the text line each block starts at, to point at code no input reached
    block_lines: Vec<u32>,
    /// the text line of each instruction
    lines: Vec<u32>,
}

// ------------------------------------------------------------------------------------------
// Parser: one pass over the lines, no backtracking beyond one token.

struct Cur<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Cur<'a> {
    fn new(line: &'a str) -> Self {
        Cur { s: line.as_bytes(), i: 0 }
    }

    /// The next byte after blanks; 0 at the end of the line or at a comment.
    fn peek(&mut self) -> u8 {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t') {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b';') | None => 0,
            Some(&b) => b,
        }
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == b {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, b: u8) -> R<()> {
        if self.eat(b) {
            Ok(())
        } else {
            Err(format!("expected '{}' at column {}", b as char, self.i + 1))
        }
    }

    fn text(&self, from: usize, to: usize) -> &'a str {
        std::str::from_utf8(&self.s[from..to]).unwrap_or("")
    }

    /// A keyword, type word or number: letters, digits and ` _ . $ - + `.
    fn word(&mut self) -> &'a str {
        self.peek();
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || matches!(self.s[self.i], b'_' | b'.' | b'$' | b'-' | b'+')) {
            self.i += 1;
        }
        self.text(start, self.i)
    }

    fn eat_word(&mut self, w: &str) -> bool {
        let save = self.i;
        if self.word() == w {
            true
        } else {
            self.i = save;
            false
        }
    }

    fn expect_word(&mut self, w: &str) -> R<()> {
        if self.eat_word(w) {
            Ok(())
        } else {
            Err(format!("expected '{w}' at column {}", self.i + 1))
        }
    }

    /// The name after a ` % ` or ` @ ` sigil, plain or quoted (escapes are kept as written).
    fn name(&mut self) -> R<&'a str> {
        let start = self.i;
        if self.s.get(self.i) == Some(&b'"') {
            let mut j = start + 1;
            while j < self.s.len() && self.s[j] != b'"' {
                j += 1;
            }
            if j >= self.s.len() {
                return Err("unterminated quoted name".into());
            }
            self.i = j + 1;
            return Ok(self.text(start + 1, j));
        }
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || matches!(self.s[self.i], b'_' | b'.' | b'$' | b'-')) {
            self.i += 1;
        }
        if self.i == start {
            return Err(format!("expected a name at column {}", start + 1));
        }
        Ok(self.text(start, self.i))
    }

    fn uint(&mut self) -> R<u64> {
        let w = self.word();
        w.parse::<u64>().map_err(|_| format!("expected a number, found '{w}'"))
    }
}

fn parse_int(w: &str) -> R<u64> {
    let (neg, digits) = match w.strip_prefix('-') {
        Some(d) => (true, d),
        None => (false, w),
    };
    let v = digits.parse::<u128>().map_err(|_| format!("expected an integer, found '{w}'"))? as u64;
    Ok(if neg { v.wrapping_neg() } else { v })
}

/// A float literal as LLVM prints it: a decimal that is exact for the type, or the hex of
/// the double with the same value (`0xH` and four digits for half).
fn parse_fp(w: &str, bits: u8) -> R<u64> {
    if let Some(h) = w.strip_prefix("0xH") {
        return u64::from_str_radix(h, 16).map_err(|_| format!("bad half literal '{w}'"));
    }
    let d = match w.strip_prefix("0x") {
        Some(h) => f64::from_bits(u64::from_str_radix(h, 16).map_err(|_| format!("bad float literal '{w}'"))?),
        None => w.parse::<f64>().map_err(|_| format!("bad float literal '{w}'"))?,
    };
    Ok(match bits {
        64 => d.to_bits(),
        32 if d.is_nan() => {
            let b = d.to_bits();
            ((b >> 32) & 0x8000_0000) | 0x7F80_0000 | ((b >> 29) & 0x7F_FFFF)
        }
        32 => (d as f32).to_bits() as u64,
        _ => f64_to_half(d) as u64,
    })
}

/// How a groupshared global is called in the source: DXC prints the mangled name
/// (`\01?name@@3IA`), we may print either.
fn source_name(global: &str) -> &str {
    let n = global.strip_prefix("\\01").unwrap_or(global);
    match n.strip_prefix('?') {
        Some(rest) => rest.split('@').next().unwrap_or(rest),
        None => n,
    }
}

struct Parser {
    types: Types,
    m: Module,
    /// name of a global → its pointer
    globals: HashMap<String, u64>,
    /// `%N` → register
    num: Vec<u32>,
    named: HashMap<String, u32>,
    /// register → block, for the registers that are labels
    label_block: Vec<u32>,
    name_ids: HashMap<String, u32>,
    /// the switch whose case lines are being read
    open_switch: Option<usize>,
    in_func: bool,
    defined: bool,
    dict: std::collections::BTreeSet<u32>,
    line_no: u32,
}

const MAX_CELLS: u64 = 1 << 26;

impl Parser {
    fn new() -> Self {
        let m = Module {
            insts: Vec::new(),
            blocks: Vec::new(),
            phi_in: Vec::new(),
            cases: Vec::new(),
            args: Vec::new(),
            gep: Vec::new(),
            nregs: 0,
            aggs: vec![[0; 5]],
            regions: vec![Region { start: 0, len: 0, shared: NONE }],
            mem: Vec::new(),
            names: Vec::new(),
            shared: Vec::new(),
            dict: Vec::new(),
            block_lines: Vec::new(),
            lines: Vec::new(),
        };
        Parser { types: Types::new(), m, globals: HashMap::new(), num: Vec::new(), named: HashMap::new(), label_block: Vec::new(), name_ids: HashMap::new(), open_switch: None, in_func: false, defined: false, dict: Default::default(), line_no: 0 }
    }

    fn new_reg(&mut self) -> u32 {
        self.m.nregs += 1;
        self.label_block.push(NONE);
        self.m.nregs - 1
    }

    fn local(&mut self, name: &str) -> u32 {
        let b = name.as_bytes();
        if b.len() <= 9 && b.iter().all(|c| c.is_ascii_digit()) {
            let n = b.iter().fold(0usize, |n, &d| n * 10 + (d - b'0') as usize);
            if n >= self.num.len() {
                self.num.resize((n + 1).max(self.num.len() * 2), NONE);
            }
            if self.num[n] == NONE {
                self.num[n] = self.new_reg();
            }
            return self.num[n];
        }
        if let Some(&r) = self.named.get(name) {
            return r;
        }
        let r = self.new_reg();
        self.named.insert(name.to_string(), r);
        r
    }

    fn compared(&mut self, w: u8, a: Opd, b: Opd) {
        for o in [a, b] {
            if let (Opd::Imm(v), true, true) = (o, w <= 32, self.dict.len() < 4096) {
                self.dict.insert(v as u32);
            }
        }
    }

    fn name_id(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.name_ids.get(name) {
            return i;
        }
        self.m.names.push(name.to_string());
        let i = (self.m.names.len() - 1) as u32;
        self.name_ids.insert(name.to_string(), i);
        i
    }

    fn region(&mut self, cells: u64, shared: u32) -> R<u64> {
        if self.m.mem.len() as u64 + cells > MAX_CELLS {
            return Err("the program's memory is larger than the interpreter allows".into());
        }
        let start = self.m.mem.len() as u32;
        self.m.mem.resize(self.m.mem.len() + cells as usize, 0);
        self.m.regions.push(Region { start, len: cells as u32, shared });
        Ok(((self.m.regions.len() - 1) as u64) << 32)
    }

    fn ty(&mut self, c: &mut Cur) -> R<TyId> {
        let mut t = match c.peek() {
            b'[' => {
                c.i += 1;
                let n = c.uint()?;
                c.expect_word("x")?;
                let of = self.ty(c)?;
                c.expect(b']')?;
                self.types.array(u32::try_from(n).map_err(|_| "array too long".to_string())?, of)
            }
            b'{' => {
                c.i += 1;
                let mut fields = Vec::new();
                while c.peek() != b'}' {
                    fields.push(self.ty(c)?);
                    if !c.eat(b',') {
                        break;
                    }
                }
                c.expect(b'}')?;
                self.types.add(TyDef::Struct(fields))
            }
            b'%' => {
                c.i += 1;
                let n = c.name()?;
                self.types.named(n)
            }
            b'<' => return Err("vector types are not part of the subset".into()),
            _ => match c.word() {
                "void" => T_VOID,
                "i1" => T_I1,
                "i8" => T_I8,
                "i16" => T_I16,
                "i32" => T_I32,
                "i64" => T_I64,
                "half" => T_HALF,
                "float" => T_FLOAT,
                "double" => T_DOUBLE,
                "label" => T_LABEL,
                "metadata" => T_METADATA,
                w => match w.strip_prefix('i').and_then(|b| b.parse::<u8>().ok()) {
                    Some(b) if (1..=64).contains(&b) => self.types.add(TyDef::Int(b)),
                    _ => return Err(format!("expected a type, found '{w}' at column {}", c.i + 1)),
                },
            },
        };
        loop {
            match c.peek() {
                b'*' => {
                    c.i += 1;
                    t = self.types.ptr(t);
                }
                b'a' if c.s[c.i..].starts_with(b"addrspace(") => {
                    while c.i < c.s.len() && c.s[c.i] != b')' {
                        c.i += 1;
                    }
                    c.i += 1;
                }
                _ => return Ok(t),
            }
        }
    }

    /// The cell offset an index list reaches from a pointer to `src`, for constant indices.
    /// `each` gets (index operand, stride) for the indices that are not constants.
    fn gep_walk(&mut self, c: &mut Cur, src: TyId, mut each: impl FnMut(Opd, i32, u8)) -> R<i32> {
        let mut cur = src;
        let mut off = 0i64;
        let mut first = true;
        while c.eat(b',') {
            if c.peek() == b'!' {
                break;
            }
            let it = self.ty(c)?;
            let w = self.types.int_bits(it)?;
            let idx = self.operand(c, it)?;
            let konst = match idx {
                Opd::Imm(v) => Some(sx(v, w)),
                Opd::Reg(_) => None,
            };
            let stride;
            if first {
                stride = self.types.cells(cur) as i64;
                first = false;
            } else {
                match self.types.defs[cur as usize].clone() {
                    TyDef::Array(_, of) => {
                        stride = self.types.cells(of) as i64;
                        cur = of;
                    }
                    TyDef::Struct(fields) => {
                        let k = konst.ok_or("a struct index must be a constant")? as usize;
                        if k >= fields.len() {
                            return Err("struct index out of range".into());
                        }
                        off += fields[..k].iter().map(|f| self.types.cells(*f) as i64).sum::<i64>();
                        cur = fields[k];
                        continue;
                    }
                    other => return Err(format!("getelementptr into {other:?}")),
                }
            }
            match konst {
                Some(k) => off = off.wrapping_add(k.wrapping_mul(stride)),
                None => each(idx, stride as i32, w),
            }
        }
        Ok(off as i32)
    }

    /// A value of type `t`: a register or a constant.
    fn operand(&mut self, c: &mut Cur, t: TyId) -> R<Opd> {
        match c.peek() {
            b'%' => {
                c.i += 1;
                let n = c.name()?;
                Ok(Opd::Reg(self.local(n)))
            }
            b'@' => {
                c.i += 1;
                let n = c.name()?;
                self.globals.get(n).map(|&p| Opd::Imm(p)).ok_or_else(|| format!("unknown global @{n}"))
            }
            b'{' => {
                c.i += 1;
                let mut v = [0u64; 5];
                let mut k = 0;
                while c.peek() != b'}' {
                    let ft = self.ty(c)?;
                    let Opd::Imm(x) = self.operand(c, ft)? else { return Err("a struct constant with a non-constant field".into()) };
                    if k >= 5 || self.types.is_agg(ft) {
                        return Err("a struct constant the interpreter cannot hold".into());
                    }
                    v[k] = x;
                    k += 1;
                    if !c.eat(b',') {
                        break;
                    }
                }
                c.expect(b'}')?;
                self.m.aggs.push(v);
                Ok(Opd::Imm((self.m.aggs.len() - 1) as u64))
            }
            _ => {
                let w = c.word();
                match w {
                    "true" => Ok(Opd::Imm(1)),
                    "false" | "undef" | "null" | "zeroinitializer" => Ok(Opd::Imm(0)),
                    "getelementptr" => {
                        c.eat_word("inbounds");
                        c.expect(b'(')?;
                        let src = self.ty(c)?;
                        c.expect(b',')?;
                        let bt = self.ty(c)?;
                        let Opd::Imm(base) = self.operand(c, bt)? else { return Err("a constant getelementptr on a register".into()) };
                        let off = self.gep_walk(c, src, |_, _, _| {})?;
                        c.expect(b')')?;
                        Ok(Opd::Imm((base & !0xFFFF_FFFF) | (base as u32).wrapping_add(off as u32) as u64))
                    }
                    "bitcast" | "addrspacecast" => {
                        c.expect(b'(')?;
                        let bt = self.ty(c)?;
                        let v = self.operand(c, bt)?;
                        c.expect_word("to")?;
                        self.ty(c)?;
                        c.expect(b')')?;
                        Ok(v)
                    }
                    "" => Err(format!("expected a value at column {}", c.i + 1)),
                    _ => match self.types.defs[t as usize] {
                        TyDef::Int(b) => Ok(Opd::Imm(parse_int(w)? & mask(b))),
                        TyDef::Half => Ok(Opd::Imm(parse_fp(w, 16)?)),
                        TyDef::Float => Ok(Opd::Imm(parse_fp(w, 32)?)),
                        TyDef::Double => Ok(Opd::Imm(parse_fp(w, 64)?)),
                        _ => Err(format!("unexpected constant '{w}'")),
                    },
                }
            }
        }
    }

    /// The cells of a global's initialiser.
    fn init_cells(&mut self, c: &mut Cur, t: TyId, out: &mut Vec<u64>) -> R<()> {
        let n = self.types.cells(t);
        if matches!(c.peek(), b'z' | b'u') {
            let save = c.i;
            if matches!(c.word(), "zeroinitializer" | "undef") {
                if out.len() as u64 + n > MAX_CELLS {
                    return Err("initialiser too large".into());
                }
                out.resize(out.len() + n as usize, 0);
                return Ok(());
            }
            c.i = save;
        }
        let close = match (c.peek(), self.types.is_agg(t)) {
            (b'[', true) => b']',
            (b'{', true) => b'}',
            (_, true) => return Err("unsupported aggregate initialiser".into()),
            _ => {
                let Opd::Imm(v) = self.operand(c, t)? else { return Err("an initialiser must be constant".into()) };
                out.push(v);
                return Ok(());
            }
        };
        c.i += 1;
        while c.peek() != close {
            let et = self.ty(c)?;
            self.init_cells(c, et, out)?;
            if !c.eat(b',') {
                break;
            }
        }
        c.expect(close)
    }

    fn type_def(&mut self, c: &mut Cur) -> R<()> {
        c.expect(b'%')?;
        let name = c.name()?;
        c.expect(b'=')?;
        c.expect_word("type")?;
        if name == "dx.types.Handle" || c.peek() != b'{' {
            return Ok(());
        }
        let id = self.types.named(name);
        let body = self.ty(c)?;
        self.types.defs[id as usize] = self.types.defs[body as usize].clone();
        Ok(())
    }

    fn global(&mut self, c: &mut Cur) -> R<()> {
        c.expect(b'@')?;
        let name = c.name()?;
        c.expect(b'=')?;
        let mut shared = false;
        loop {
            match c.word() {
                "global" | "constant" => break,
                "addrspace" => {
                    c.expect(b'(')?;
                    shared = c.uint()? == 3;
                    c.expect(b')')?;
                }
                "" => return Err("expected 'global' or 'constant'".into()),
                _ => {}
            }
        }
        let t = self.ty(c)?;
        let cells = self.types.cells(t);
        let shared_id = if shared {
            self.m.shared.push(source_name(name).to_string());
            (self.m.shared.len() - 1) as u32
        } else {
            NONE
        };
        let p = self.region(cells, shared_id)?;
        if !matches!(c.peek(), 0 | b',') {
            let mut init = Vec::new();
            self.init_cells(c, t, &mut init)?;
            if init.len() as u64 != cells {
                return Err(format!("the initialiser of @{name} has {} cells, its type {cells}", init.len()));
            }
            let start = self.m.regions[(p >> 32) as usize].start as usize;
            self.m.mem[start..start + init.len()].copy_from_slice(&init);
        }
        self.globals.insert(name.to_string(), p);
        Ok(())
    }

    fn block(&mut self, label: u32) {
        // a label line before any instruction names the entry block
        if !(self.m.blocks.len() == 1 && self.m.insts.is_empty()) {
            self.m.blocks.push(self.m.insts.len() as u32);
            self.m.block_lines.push(self.line_no);
        }
        self.label_block[label as usize] = (self.m.blocks.len() - 1) as u32;
    }

    fn label(&mut self, c: &mut Cur) -> R<u32> {
        c.expect_word("label")?;
        c.expect(b'%')?;
        let n = c.name()?;
        Ok(self.local(n))
    }

    fn skip_words(c: &mut Cur, words: &[&str]) {
        loop {
            let save = c.i;
            if !words.contains(&c.word()) {
                c.i = save;
                return;
            }
        }
    }

    fn case_lines(&mut self, c: &mut Cur, at: usize) -> R<()> {
        loop {
            if c.eat(b']') {
                self.open_switch = None;
                return Ok(());
            }
            if c.peek() == 0 {
                self.open_switch = Some(at);
                return Ok(());
            }
            let t = self.ty(c)?;
            let Opd::Imm(v) = self.operand(c, t)? else { return Err("a switch case must be a constant".into()) };
            c.expect(b',')?;
            let to = self.label(c)?;
            self.m.cases.push((v, to));
            self.compared(32, Opd::Imm(v & mask(32)), Opd::Reg(0));
            if let Inst::Switch { n, .. } = &mut self.m.insts[at] {
                *n += 1;
            }
        }
    }

    fn inst(&mut self, c: &mut Cur) -> R<()> {
        let mut dst = NONE;
        if c.peek() == b'%' {
            c.i += 1;
            let n = c.name()?;
            dst = self.local(n);
            c.expect(b'=')?;
        }
        let mut op = c.word();
        if matches!(op, "tail" | "musttail" | "notail") {
            op = c.word();
        }
        let bin = |op| Some(match op {
            "add" => BinOp::Add,
            "sub" => BinOp::Sub,
            "mul" => BinOp::Mul,
            "udiv" => BinOp::UDiv,
            "sdiv" => BinOp::SDiv,
            "urem" => BinOp::URem,
            "srem" => BinOp::SRem,
            "shl" => BinOp::Shl,
            "lshr" => BinOp::LShr,
            "ashr" => BinOp::AShr,
            "and" => BinOp::And,
            "or" => BinOp::Or,
            "xor" => BinOp::Xor,
            _ => return None,
        });
        let fbin = |op| Some(match op {
            "fadd" => FOp::Add,
            "fsub" => FOp::Sub,
            "fmul" => FOp::Mul,
            "fdiv" => FOp::Div,
            "frem" => FOp::Rem,
            _ => return None,
        });
        const FAST: [&str; 6] = ["fast", "nnan", "ninf", "nsz", "arcp", "contract"];
        let inst = if let Some(op) = bin(op) {
            Self::skip_words(c, &["nuw", "nsw", "exact"]);
            let t = self.ty(c)?;
            let w = self.types.int_bits(t)?;
            let a = self.operand(c, t)?;
            c.expect(b',')?;
            let b = self.operand(c, t)?;
            Inst::Bin { op, w, dst, a, b }
        } else if let Some(op) = fbin(op) {
            Self::skip_words(c, &FAST);
            let t = self.ty(c)?;
            let w = self.types.fp_bits(t)?;
            let a = self.operand(c, t)?;
            c.expect(b',')?;
            let b = self.operand(c, t)?;
            Inst::FBin { op, w, dst, a, b }
        } else {
            match op {
                "icmp" => {
                    let pred = match c.word() {
                        "eq" => IPred::Eq,
                        "ne" => IPred::Ne,
                        "ugt" => IPred::Ugt,
                        "uge" => IPred::Uge,
                        "ult" => IPred::Ult,
                        "ule" => IPred::Ule,
                        "sgt" => IPred::Sgt,
                        "sge" => IPred::Sge,
                        "slt" => IPred::Slt,
                        "sle" => IPred::Sle,
                        p => return Err(format!("icmp predicate '{p}'")),
                    };
                    let t = self.ty(c)?;
                    let w = self.types.int_bits(t)?;
                    let a = self.operand(c, t)?;
                    c.expect(b',')?;
                    let b = self.operand(c, t)?;
                    self.compared(w, a, b);
                    Inst::ICmp { pred, w, dst, a, b }
                }
                "fcmp" => {
                    Self::skip_words(c, &FAST);
                    let pred = match c.word() {
                        "false" => FPred::False,
                        "oeq" => FPred::Oeq,
                        "ogt" => FPred::Ogt,
                        "oge" => FPred::Oge,
                        "olt" => FPred::Olt,
                        "ole" => FPred::Ole,
                        "one" => FPred::One,
                        "ord" => FPred::Ord,
                        "ueq" => FPred::Ueq,
                        "ugt" => FPred::Ugt,
                        "uge" => FPred::Uge,
                        "ult" => FPred::Ult,
                        "ule" => FPred::Ule,
                        "une" => FPred::Une,
                        "uno" => FPred::Uno,
                        "true" => FPred::True,
                        p => return Err(format!("fcmp predicate '{p}'")),
                    };
                    let t = self.ty(c)?;
                    let w = self.types.fp_bits(t)?;
                    let a = self.operand(c, t)?;
                    c.expect(b',')?;
                    let b = self.operand(c, t)?;
                    self.compared(w, a, b);
                    Inst::FCmp { pred, w, dst, a, b }
                }
                "trunc" | "zext" | "sext" | "fptoui" | "fptosi" | "uitofp" | "sitofp" | "fptrunc" | "fpext" | "bitcast" | "addrspacecast" | "ptrtoint" | "inttoptr" => {
                    let ft = self.ty(c)?;
                    let a = self.operand(c, ft)?;
                    c.expect_word("to")?;
                    let tt = self.ty(c)?;
                    let (ib, fb) = (|p: &Parser, t| p.types.int_bits(t), |p: &Parser, t| p.types.fp_bits(t));
                    match op {
                        "trunc" => Inst::Cast { op: CastOp::Trunc, from: ib(self, ft)?, to: ib(self, tt)?, dst, a },
                        "sext" => Inst::Cast { op: CastOp::SExt, from: ib(self, ft)?, to: ib(self, tt)?, dst, a },
                        "fptoui" => Inst::Cast { op: CastOp::FpToUi, from: fb(self, ft)?, to: ib(self, tt)?, dst, a },
                        "fptosi" => Inst::Cast { op: CastOp::FpToSi, from: fb(self, ft)?, to: ib(self, tt)?, dst, a },
                        "uitofp" => Inst::Cast { op: CastOp::UiToFp, from: ib(self, ft)?, to: fb(self, tt)?, dst, a },
                        "sitofp" => Inst::Cast { op: CastOp::SiToFp, from: ib(self, ft)?, to: fb(self, tt)?, dst, a },
                        "fptrunc" | "fpext" => Inst::Cast { op: CastOp::FpConv, from: fb(self, ft)?, to: fb(self, tt)?, dst, a },
                        _ => Inst::Copy { dst, a },
                    }
                }
                "select" => {
                    let ct = self.ty(c)?;
                    let cond = self.operand(c, ct)?;
                    c.expect(b',')?;
                    let t = self.ty(c)?;
                    let a = self.operand(c, t)?;
                    c.expect(b',')?;
                    let t = self.ty(c)?;
                    let b = self.operand(c, t)?;
                    Inst::Select { dst, c: cond, a, b }
                }
                "phi" => {
                    let t = self.ty(c)?;
                    let first = self.m.phi_in.len() as u32;
                    while c.eat(b'[') {
                        let v = self.operand(c, t)?;
                        c.expect(b',')?;
                        c.expect(b'%')?;
                        let n = c.name()?;
                        let from = self.local(n);
                        c.expect(b']')?;
                        self.m.phi_in.push((v, from));
                        if !c.eat(b',') {
                            break;
                        }
                    }
                    Inst::Phi { dst, first, n: self.m.phi_in.len() as u32 - first }
                }
                "br" => {
                    if c.peek_is_label() {
                        Inst::Br { to: self.label(c)? }
                    } else {
                        let t = self.ty(c)?;
                        let cond = self.operand(c, t)?;
                        c.expect(b',')?;
                        let t = self.label(c)?;
                        c.expect(b',')?;
                        let f = self.label(c)?;
                        Inst::CondBr { c: cond, t, f }
                    }
                }
                "switch" => {
                    let t = self.ty(c)?;
                    let v = self.operand(c, t)?;
                    c.expect(b',')?;
                    let default = self.label(c)?;
                    c.expect(b'[')?;
                    self.m.insts.push(Inst::Switch { v, default, first: self.m.cases.len() as u32, n: 0 });
                    self.m.lines.push(self.line_no);
                    return self.case_lines(c, self.m.insts.len() - 1);
                }
                "ret" => Inst::Ret,
                "unreachable" => Inst::Unreachable,
                "call" => {
                    let rt = self.ty(c)?;
                    c.expect(b'@')?;
                    let callee = c.name()?;
                    c.expect(b'(')?;
                    let first = self.m.args.len() as u32;
                    while c.peek() != b')' {
                        let t = self.ty(c)?;
                        Self::skip_words(c, &["nocapture", "readonly", "readnone", "noalias", "nonnull", "zeroext", "signext"]);
                        let a = self.operand(c, t)?;
                        self.m.args.push(a);
                        if !c.eat(b',') {
                            break;
                        }
                    }
                    c.expect(b')')?;
                    let n = self.m.args.len() as u32 - first;
                    if callee.starts_with("llvm.lifetime.") || callee.starts_with("llvm.dbg.") {
                        self.m.args.truncate(first as usize);
                        return Ok(());
                    }
                    let name = self.name_id(callee);
                    match (callee.strip_prefix("dx.op."), self.m.args.get(first as usize)) {
                        (Some(rest), Some(&Opd::Imm(opcode))) if opcode < 1 << 16 => {
                            let ovl = match rest.rsplit('.').next().unwrap_or("") {
                                "i1" => 1,
                                "i8" => 8,
                                "i16" => 16,
                                "i32" => 32,
                                "i64" => 64,
                                "f16" => OV_F16,
                                "f32" => OV_F32,
                                "f64" => OV_F64,
                                _ => 0,
                            };
                            let agg = if self.types.is_agg(rt) {
                                self.m.aggs.push([0; 5]);
                                (self.m.aggs.len() - 1) as u32
                            } else {
                                NONE
                            };
                            Inst::DxOp { op: opcode as u16, ovl, dst, first, n, agg, name }
                        }
                        _ => {
                            self.m.args.truncate(first as usize);
                            Inst::Unknown { name }
                        }
                    }
                }
                "extractvalue" => {
                    let t = self.ty(c)?;
                    let a = self.operand(c, t)?;
                    c.expect(b',')?;
                    let idx = c.uint()?;
                    if idx >= 5 || c.peek() == b',' && !c.s[c.i..].starts_with(b", !") {
                        return Err("extractvalue beyond what the interpreter holds".into());
                    }
                    Inst::ExtractValue { dst, a, idx: idx as u8 }
                }
                "load" => {
                    Self::skip_words(c, &["volatile", "atomic"]);
                    let t = self.ty(c)?;
                    if self.types.is_agg(t) {
                        return Err("a load of an aggregate".into());
                    }
                    c.expect(b',')?;
                    let pt = self.ty(c)?;
                    let p = self.operand(c, pt)?;
                    Inst::Load { dst, p }
                }
                "store" => {
                    Self::skip_words(c, &["volatile", "atomic"]);
                    let t = self.ty(c)?;
                    if self.types.is_agg(t) {
                        return Err("a store of an aggregate".into());
                    }
                    let v = self.operand(c, t)?;
                    c.expect(b',')?;
                    let pt = self.ty(c)?;
                    let p = self.operand(c, pt)?;
                    Inst::Store { v, p }
                }
                "getelementptr" => {
                    c.eat_word("inbounds");
                    let src = self.ty(c)?;
                    c.expect(b',')?;
                    let bt = self.ty(c)?;
                    let base = self.operand(c, bt)?;
                    let first = self.m.gep.len() as u32;
                    let mut dynamic = Vec::new();
                    let off = self.gep_walk(c, src, |o, s, w| dynamic.push((o, s, w)))?;
                    self.m.gep.extend(dynamic);
                    Inst::Gep { dst, base, off, first, n: self.m.gep.len() as u32 - first }
                }
                "alloca" => {
                    let t = self.ty(c)?;
                    if c.eat(b',') && !c.eat_word("align") {
                        return Err("an alloca with a count".into());
                    }
                    let cells = self.types.cells(t);
                    Inst::Alloca { dst, ptr: self.region(cells, NONE)? }
                }
                "atomicrmw" => {
                    c.eat_word("volatile");
                    let op = match c.word() {
                        "xchg" => RmwOp::Xchg,
                        "add" => RmwOp::Add,
                        "sub" => RmwOp::Sub,
                        "and" => RmwOp::And,
                        "nand" => RmwOp::Nand,
                        "or" => RmwOp::Or,
                        "xor" => RmwOp::Xor,
                        "max" => RmwOp::Max,
                        "min" => RmwOp::Min,
                        "umax" => RmwOp::UMax,
                        "umin" => RmwOp::UMin,
                        o => return Err(format!("atomicrmw operation '{o}'")),
                    };
                    let pt = self.ty(c)?;
                    let p = self.operand(c, pt)?;
                    c.expect(b',')?;
                    let t = self.ty(c)?;
                    let w = self.types.int_bits(t)?;
                    let v = self.operand(c, t)?;
                    Inst::AtomicRmw { op, w, dst, p, v }
                }
                "cmpxchg" => {
                    Self::skip_words(c, &["weak", "volatile"]);
                    let pt = self.ty(c)?;
                    let p = self.operand(c, pt)?;
                    c.expect(b',')?;
                    let t = self.ty(c)?;
                    let cmp = self.operand(c, t)?;
                    c.expect(b',')?;
                    let t = self.ty(c)?;
                    let new = self.operand(c, t)?;
                    self.m.aggs.push([0; 5]);
                    Inst::CmpXchg { dst, agg: (self.m.aggs.len() - 1) as u32, p, cmp, new }
                }
                other => return Err(format!("instruction '{other}' is not part of the subset")),
            }
        };
        self.m.insts.push(inst);
        self.m.lines.push(self.line_no);
        Ok(())
    }

    fn line(&mut self, line: &str) -> R<()> {
        let mut c = Cur::new(line);
        if !self.in_func {
            return match line.as_bytes().first() {
                Some(b'%') => self.type_def(&mut c),
                Some(b'@') => self.global(&mut c),
                Some(b'd') if line.starts_with("define ") => {
                    if self.defined {
                        return Err("more than one function is defined (only a fully inlined entry point is supported)".into());
                    }
                    if !line.contains("()") {
                        return Err("the entry function takes parameters".into());
                    }
                    self.defined = true;
                    self.in_func = true;
                    self.m.blocks.push(0);
                    self.m.block_lines.push(self.line_no);
                    let entry = self.local("0");
                    self.label_block[entry as usize] = 0;
                    Ok(())
                }
                _ => Ok(()),
            };
        }
        if let Some(at) = self.open_switch {
            return self.case_lines(&mut c, at);
        }
        match line.as_bytes().first() {
            None => Ok(()),
            Some(b'}') => {
                self.in_func = false;
                Ok(())
            }
            Some(b' ') | Some(b'\t') => {
                if c.peek() == 0 {
                    return Ok(());
                }
                self.inst(&mut c)
            }
            Some(b';') => {
                if let Some(rest) = line.strip_prefix("; <label>:") {
                    let n = rest.split(|ch: char| !ch.is_ascii_digit()).next().unwrap_or("");
                    let label = self.local(n);
                    self.block(label);
                }
                Ok(())
            }
            Some(_) => {
                let name = if line.starts_with('"') { c.name()? } else { line.split(':').next().unwrap_or("") };
                let label = self.local(name);
                self.block(label);
                Ok(())
            }
        }
    }

    /// Labels were read as registers; turn them into block numbers.
    fn finish(mut self) -> R<Module> {
        if !self.defined || self.in_func {
            return Err("no complete function definition".into());
        }
        let lb = &self.label_block;
        let block = |r: u32| match lb.get(r as usize) {
            Some(&b) if b != NONE => Ok(b),
            _ => Err("a branch to a label that is not defined".to_string()),
        };
        for inst in &mut self.m.insts {
            match inst {
                Inst::Br { to } => *to = block(*to)?,
                Inst::CondBr { t, f, .. } => {
                    *t = block(*t)?;
                    *f = block(*f)?;
                }
                Inst::Switch { default, .. } => *default = block(*default)?,
                _ => {}
            }
        }
        for (_, b) in &mut self.m.phi_in {
            // a phi may name a predecessor that was never printed (dead); it is never taken
            *b = block(*b).unwrap_or(NONE);
        }
        for (_, b) in &mut self.m.cases {
            *b = block(*b)?;
        }
        self.m.dict = self.dict.iter().copied().collect();
        Ok(self.m)
    }
}

impl Cur<'_> {
    fn peek_is_label(&mut self) -> bool {
        self.peek();
        self.s[self.i..].starts_with(b"label ")
    }
}

fn parse_module(text: &str) -> R<Module> {
    let mut p = Parser::new();
    for (n, line) in text.lines().enumerate() {
        p.line_no = n as u32 + 1;
        p.line(line).map_err(|e| format!("line {}: {e}: {}", n + 1, line.chars().take(160).collect::<String>()))?;
    }
    p.finish()
}

// ------------------------------------------------------------------------------------------
// Scalar semantics. Integers are kept zero-extended in a u64; floats as their bits.

fn mask(w: u8) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

fn sx(v: u64, w: u8) -> i64 {
    let s = 64 - (w.clamp(1, 64) as u32);
    ((v << s) as i64) >> s
}

/// LLVM leaves division by zero and shifts by the width or more undefined; here they have
/// one fixed result (what D3D hardware gives for division: all ones, the dividend for the
/// remainder), the same for both programs.
fn int_bin(op: BinOp, w: u8, a: u64, b: u64) -> u64 {
    let m = mask(w);
    let (x, y) = (sx(a, w), sx(b, w));
    let r = match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::UDiv => {
            if b == 0 {
                m
            } else {
                a / b
            }
        }
        BinOp::SDiv => {
            if y == 0 {
                m
            } else {
                x.wrapping_div(y) as u64
            }
        }
        BinOp::URem => {
            if b == 0 {
                a
            } else {
                a % b
            }
        }
        BinOp::SRem => {
            if y == 0 {
                a
            } else {
                x.wrapping_rem(y) as u64
            }
        }
        BinOp::Shl => {
            if b >= w as u64 {
                0
            } else {
                a << b
            }
        }
        BinOp::LShr => {
            if b >= w as u64 {
                0
            } else {
                a >> b
            }
        }
        BinOp::AShr => (x >> b.min(63)) as u64,
        BinOp::And => a & b,
        BinOp::Or => a | b,
        BinOp::Xor => a ^ b,
    };
    r & m
}

fn int_cmp(pred: IPred, w: u8, a: u64, b: u64) -> bool {
    let (x, y) = (sx(a, w), sx(b, w));
    match pred {
        IPred::Eq => a == b,
        IPred::Ne => a != b,
        IPred::Ugt => a > b,
        IPred::Uge => a >= b,
        IPred::Ult => a < b,
        IPred::Ule => a <= b,
        IPred::Sgt => x > y,
        IPred::Sge => x >= y,
        IPred::Slt => x < y,
        IPred::Sle => x <= y,
    }
}

fn half_to_f64(h: u16) -> f64 {
    let (e, m) = (((h >> 10) & 0x1F) as i32, (h & 0x3FF) as f64);
    let v = match e {
        0 => m * 2f64.powi(-24),
        31 if m == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1024.0 + m) * 2f64.powi(e - 25),
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}

/// Round to nearest, ties to even.
fn f64_to_half(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    if v.is_nan() {
        return sign | 0x7E00;
    }
    let a = v.abs();
    if a >= 65520.0 {
        return sign | 0x7C00;
    }
    if a < 2f64.powi(-14) {
        return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
    }
    let e = ((a.to_bits() >> 52) as i32) - 1023;
    let r = (a * 2f64.powi(10 - e)).round_ties_even() as u32;
    let (e, r) = if r == 2048 { (e + 1, 1024) } else { (e, r) };
    sign | (((e + 15) as u16) << 10) | (r - 1024) as u16
}

fn fp_is_nan(bits: u64, w: u8) -> bool {
    match w {
        16 => bits & 0x7C00 == 0x7C00 && bits & 0x3FF != 0,
        32 => f32::from_bits(bits as u32).is_nan(),
        _ => f64::from_bits(bits).is_nan(),
    }
}

fn fp_quiet(bits: u64, w: u8) -> u64 {
    bits | match w {
        16 => 0x200,
        32 => 0x40_0000,
        _ => 0x8_0000_0000_0000,
    }
}

fn fp_default_nan(w: u8) -> u64 {
    match w {
        16 => 0x7E00,
        32 => 0x7FC0_0000,
        _ => 0x7FF8_0000_0000_0000,
    }
}

fn to_f64(bits: u64, w: u8) -> f64 {
    match w {
        16 => half_to_f64(bits as u16),
        32 => f32::from_bits(bits as u32) as f64,
        _ => f64::from_bits(bits),
    }
}

fn from_f64(v: f64, w: u8) -> u64 {
    match w {
        16 => f64_to_half(v) as u64,
        32 => (v as f32).to_bits() as u64,
        _ => v.to_bits(),
    }
}

/// A NaN operand comes out quieted (the first one wins); an invalid operation gives the
/// positive default NaN, as LLVM's constant folder does. This keeps a result DXC folded at
/// compile time equal to the one computed here, and independent of the host.
fn nan_rule(w: u8, operands: &[u64], result: u64) -> u64 {
    for &o in operands {
        if fp_is_nan(o, w) {
            return fp_quiet(o, w);
        }
    }
    if fp_is_nan(result, w) {
        fp_default_nan(w)
    } else {
        result
    }
}

fn fp_bin(op: FOp, w: u8, a: u64, b: u64) -> u64 {
    // DXC's optimiser swaps the operands of fadd and fmul freely (LLVM orders them by
    // complexity), which IEEE allows: only the payload of NaN op NaN depends on the order.
    // That one case is made independent of it, so a swap is not reported as a difference.
    if matches!(op, FOp::Add | FOp::Mul) && fp_is_nan(a, w) && fp_is_nan(b, w) {
        return fp_quiet(a.max(b), w);
    }
    let r = match w {
        32 => {
            let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            (match op {
                FOp::Add => x + y,
                FOp::Sub => x - y,
                FOp::Mul => x * y,
                FOp::Div => x / y,
                FOp::Rem => x % y,
            })
            .to_bits() as u64
        }
        _ => {
            let (x, y) = (to_f64(a, w), to_f64(b, w));
            from_f64(
                match op {
                    FOp::Add => x + y,
                    FOp::Sub => x - y,
                    FOp::Mul => x * y,
                    FOp::Div => x / y,
                    FOp::Rem => x % y,
                },
                w,
            )
        }
    };
    nan_rule(w, &[a, b], r)
}

fn fp_cmp(pred: FPred, w: u8, a: u64, b: u64) -> bool {
    let (x, y) = (to_f64(a, w), to_f64(b, w));
    let uno = x.is_nan() || y.is_nan();
    match pred {
        FPred::False => false,
        FPred::Oeq => x == y,
        FPred::Ogt => x > y,
        FPred::Oge => x >= y,
        FPred::Olt => x < y,
        FPred::Ole => x <= y,
        FPred::One => !uno && x != y,
        FPred::Ord => !uno,
        FPred::Ueq => uno || x == y,
        FPred::Ugt => uno || x > y,
        FPred::Uge => uno || x >= y,
        FPred::Ult => uno || x < y,
        FPred::Ule => uno || x <= y,
        FPred::Une => uno || x != y,
        FPred::Uno => uno,
        FPred::True => true,
    }
}

/// Conversions out of range saturate and NaN gives 0 (undefined in LLVM; what D3D's ftoi
/// and ftou do).
fn cast(op: CastOp, from: u8, to: u8, a: u64) -> u64 {
    match op {
        CastOp::Trunc => a & mask(to),
        CastOp::SExt => sx(a, from) as u64 & mask(to),
        CastOp::FpToUi => {
            let v = to_f64(a, from);
            (v as u64).min(mask(to))
        }
        CastOp::FpToSi => {
            let v = to_f64(a, from) as i64;
            let (lo, hi) = (sx(1u64 << (to.clamp(1, 64) - 1), to), (mask(to) >> 1) as i64);
            v.clamp(lo, hi) as u64 & mask(to)
        }
        CastOp::UiToFp => match to {
            32 => (a as f32).to_bits() as u64,
            _ => from_f64(a as f64, to),
        },
        CastOp::SiToFp => match to {
            32 => (sx(a, from) as f32).to_bits() as u64,
            _ => from_f64(sx(a, from) as f64, to),
        },
        CastOp::FpConv => {
            if fp_is_nan(a, from) && (from == 16 || to == 16) {
                return fp_default_nan(to) | if a >> (from - 1) & 1 != 0 { 1u64 << (to - 1) } else { 0 };
            }
            match (from, to) {
                (32, 64) => (f32::from_bits(a as u32) as f64).to_bits(),
                (64, 32) => (f64::from_bits(a) as f32).to_bits() as u64,
                _ => from_f64(to_f64(a, from), to),
            }
        }
    }
}

fn rmw(op: RmwOp, w: u8, old: u64, v: u64) -> u64 {
    let (x, y) = (sx(old, w), sx(v, w));
    mask(w)
        & match op {
            RmwOp::Xchg => v,
            RmwOp::Add => old.wrapping_add(v),
            RmwOp::Sub => old.wrapping_sub(v),
            RmwOp::And => old & v,
            RmwOp::Nand => !(old & v),
            RmwOp::Or => old | v,
            RmwOp::Xor => old ^ v,
            RmwOp::Max => x.max(y) as u64,
            RmwOp::Min => x.min(y) as u64,
            RmwOp::UMax => old.max(v),
            RmwOp::UMin => old.min(v),
        }
}

// ------------------------------------------------------------------------------------------
// Effects.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fx {
    RawStore,
    BufStore,
    TexStore,
    Atomic,
    AtomicCas,
    Counter,
    Barrier,
    SharedStore,
    SharedRmw,
    SharedCas,
}

/// One thing a run did that is visible outside it. `target` is the handle, or the hash of
/// the groupshared global's name; `aux` the write mask and overload, or the atomic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Effect {
    fx: Fx,
    target: u64,
    addr: [u32; 3],
    aux: u32,
    vals: [u64; 4],
}

// Handles: tag (1 binding, 2 heap) << 60 | class << 56 | kind << 48 | space << 32 | index.
const KIND_TYPED_BUFFER: u64 = 10;
const KIND_RAW_BUFFER: u64 = 11;
const KIND_STRUCTURED_BUFFER: u64 = 12;

fn handle_kind(h: u64) -> u64 {
    (h >> 48) & 0xFF
}

fn show_handle(h: u64) -> String {
    let (class, kind, space, index) = ((h >> 56) & 0xF, handle_kind(h), (h >> 32) & 0xFFFF, h as u32);
    let place = match h >> 60 {
        1 => format!("{}{index},space{space}", ["t", "u", "b", "s"].get(class as usize).copied().unwrap_or("?")),
        2 if class == 3 => format!("SamplerDescriptorHeap[{index}]"),
        2 => format!("ResourceDescriptorHeap[{index}]"),
        _ => format!("handle {h:#x}"),
    };
    format!("{place} (kind {kind})")
}

fn show_effect(e: &Effect, shared: &HashMap<u64, String>) -> String {
    let global = || shared.get(&e.target).cloned().unwrap_or_else(|| format!("{:#x}", e.target));
    let vals = |n: usize| e.vals[..n].iter().map(|v| format!("{v:#x}")).collect::<Vec<_>>().join(", ");
    const ATOMIC: [&str; 9] = ["add", "and", "or", "xor", "imin", "imax", "umin", "umax", "exchange"];
    match e.fx {
        Fx::RawStore => format!("rawBufferStore {} index {:#x} offset {:#x} mask {:#x} = [{}]", show_handle(e.target), e.addr[0], e.addr[1], e.aux & 0xFF, vals(4)),
        Fx::BufStore => format!("bufferStore {} [{:#x}, {:#x}] mask {:#x} = [{}]", show_handle(e.target), e.addr[0], e.addr[1], e.aux & 0xFF, vals(4)),
        Fx::TexStore => format!("textureStore {} [{}, {}, {}] mask {:#x} = [{}]", show_handle(e.target), e.addr[0], e.addr[1], e.addr[2], e.aux & 0xFF, vals(4)),
        Fx::Atomic => format!("atomicBinOp {} {} [{:#x}, {:#x}, {:#x}] with {:#x}", ATOMIC.get((e.aux & 0xFF) as usize).copied().unwrap_or("?"), show_handle(e.target), e.addr[0], e.addr[1], e.addr[2], e.vals[0]),
        Fx::AtomicCas => format!("atomicCompareExchange {} [{:#x}, {:#x}, {:#x}] expect {:#x} new {:#x}", show_handle(e.target), e.addr[0], e.addr[1], e.addr[2], e.vals[0], e.vals[1]),
        Fx::Counter => format!("bufferUpdateCounter {} by {}", show_handle(e.target), e.vals[0] as i8),
        Fx::Barrier => format!("barrier mode {}", e.aux),
        Fx::SharedStore => format!("store groupshared {}[{}] = {:#x}", global(), e.addr[0], e.vals[0]),
        Fx::SharedRmw => format!("atomicrmw op {} groupshared {}[{}] with {:#x}", e.aux, global(), e.addr[0], e.vals[0]),
        Fx::SharedCas => format!("cmpxchg groupshared {}[{}] expect {:#x} new {:#x}", global(), e.addr[0], e.vals[0], e.vals[1]),
    }
}

// ------------------------------------------------------------------------------------------
// The interpreter: one thread of the compute shader, one lane of its wave.

const LANE_COUNT: u32 = 32;

/// Where the thread is, chosen by the seed: the first seeds are lane 0 of thread 0 (the
/// thread that usually initialises and reports), later ones spread out.
#[derive(Clone, Copy, Debug)]
struct Place {
    lane: u32,
    thread: u32,
    group: [u32; 3],
}

fn place(seed: u64) -> Place {
    let h = hash(&[seed, 0x71AC]);
    let lane = match seed % 8 {
        0..=2 => 0,
        3 => 1,
        4 => 31,
        _ => (h % LANE_COUNT as u64) as u32,
    };
    let thread = match seed % 8 {
        6 => lane + 32,
        7 => lane + 64,
        _ => lane,
    };
    let group = if seed % 8 == 0 { [0; 3] } else { [(h >> 8) as u32 % 4, (h >> 16) as u32 % 3, (h >> 24) as u32 % 2] };
    Place { lane, thread, group }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Stop {
    Returned,
    Budget,
    Unreachable,
}

type Key = (u64, u32, u32, u32, u32);

/// What a run is given: the seed (the thread's place, and the content of everything not
/// listed) and the resource words the input search chose itself.
#[derive(Clone, Default)]
struct Input {
    seed: u64,
    over: HashMap<Key, u32>,
}

impl Input {
    fn seed(seed: u64) -> Self {
        Input { seed, over: HashMap::new() }
    }
}

/// The blocks, branches and comparison outcomes the runs so far went through; `fresh` counts
/// what a run was the first to do. Comparisons count because the dialect is mostly
/// branch-free: a lane's fate is decided by masks made of comparison results long before a
/// branch tests them.
struct Coverage {
    blocks: Vec<bool>,
    edges: Vec<bool>,
    /// per instruction: bit 0 a comparison there gave false, bit 1 true
    conds: Vec<u8>,
    fresh: u32,
}

impl Coverage {
    fn new(m: &Module) -> Self {
        Coverage { blocks: vec![false; m.blocks.len()], edges: vec![false; 1 << 18], conds: vec![0; m.insts.len()], fresh: 0 }
    }

    fn outcome(&mut self, pc: usize, v: bool) {
        let bit = 1 << v as u8;
        if self.conds[pc - 1] & bit == 0 {
            self.conds[pc - 1] |= bit;
            self.fresh += 1;
        }
    }

    fn visit(&mut self, from: u32, to: u32) {
        if let Some(b) = self.blocks.get_mut(to as usize) {
            *b = true;
        }
        let e = (mix((from as u64) << 32 | to as u64) as usize) & (self.edges.len() - 1);
        if !self.edges[e] {
            self.edges[e] = true;
            self.fresh += 1;
        }
    }

    fn blocks_hit(&self) -> usize {
        self.blocks.iter().filter(|b| **b).count()
    }
}

struct Outcome {
    trace: Vec<Effect>,
    stop: Stop,
    steps: u64,
    /// the resource words read before anything was stored there, in order, with what they held
    reads: Vec<(Key, u32)>,
    /// for the input search: the latest comparisons that have only ever gone one way
    /// (instruction, left, right)
    cmps: Vec<(u32, u32, u32)>,
}

/// The key of a groupshared cell as an input word: what the other threads left there.
fn shared_key(name: &str, cell: u32) -> Key {
    (3 << 60 | fnv(name) >> 4, cell, 0, 0, 0)
}

struct Run<'m> {
    m: &'m Module,
    seed: u64,
    over: &'m HashMap<Key, u32>,
    reads: Vec<(Key, u32)>,
    /// memory cells read or written so far (the first read of a groupshared cell is an input)
    touched: Vec<bool>,
    cmps: Vec<(u32, u32, u32)>,
    ncmps: usize,
    /// XCL_EXEC_NANS: every float operation that gave a NaN, to find where two IRs part
    nans: Option<Vec<String>>,
    at: Place,
    regs: Vec<u64>,
    aggs: Vec<[u64; 5]>,
    mem: Vec<u64>,
    /// what was stored to resources, by word
    res: HashMap<Key, u32>,
    counters: HashMap<u64, u32>,
    trace: Vec<Effect>,
}

thread_local! {
    /// The NaN log of the last run on this thread (a debugging aid, see XCL_EXEC_NANS).
    static NAN_LOG: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn run<'m>(m: &'m Module, input: &'m Input, budget: u64, cov: Option<&mut Coverage>) -> R<Outcome> {
    let seed = input.seed;
    let mut r = Run { m, seed, over: &input.over, reads: Vec::new(), touched: vec![false; m.mem.len()], cmps: Vec::new(), ncmps: 0, nans: None, at: place(seed), regs: vec![0; m.nregs as usize], aggs: m.aggs.clone(), mem: m.mem.clone(), res: HashMap::new(), counters: HashMap::new(), trace: Vec::new() };
    // groupshared memory starts as whatever the other threads left there
    for region in &m.regions {
        if region.shared != NONE {
            let name = &m.shared[region.shared as usize];
            for k in 0..region.len {
                let key = shared_key(name, k);
                r.mem[(region.start + k) as usize] = input.over.get(&key).copied().unwrap_or_else(|| content(seed, hash(&[key.0, k as u64]))) as u64;
            }
        }
    }
    if cov.is_none() && std::env::var_os("XCL_EXEC_NANS").is_some() {
        r.nans = Some(Vec::new());
    }
    let (stop, steps) = r.exec(budget, cov)?;
    if let Some(log) = &r.nans {
        NAN_LOG.with(|l| *l.borrow_mut() = log.clone());
    }
    Ok(Outcome { trace: r.trace, stop, steps, reads: r.reads, cmps: r.cmps })
}

impl Run<'_> {
    #[inline]
    fn val(&self, o: Opd) -> u64 {
        match o {
            Opd::Reg(r) => self.regs[r as usize],
            Opd::Imm(v) => v,
        }
    }

    /// Notes a comparison's outcome for the input search, and remembers the last 256
    /// comparisons that have gone only one way so far: the search tries to turn those by
    /// moving one side onto the other.
    fn compared(&mut self, cov: &mut Coverage, pc: usize, a: u64, b: u64, v: bool) {
        cov.outcome(pc, v);
        if cov.conds[pc - 1] != 3 && a != b && (a | b) >> 32 == 0 {
            let c = (pc as u32, a as u32, b as u32);
            if self.cmps.len() < 256 {
                self.cmps.push(c);
            } else {
                self.cmps[self.ncmps % 256] = c;
            }
            self.ncmps += 1;
        }
    }

    #[inline]
    fn set(&mut self, dst: u32, v: u64) {
        if dst != NONE {
            self.regs[dst as usize] = v;
        }
    }

    fn cell(&self, p: u64) -> Option<(usize, Region)> {
        let r = *self.m.regions.get((p >> 32) as usize)?;
        ((p as u32) < r.len).then(|| ((r.start + p as u32) as usize, r))
    }

    fn load(&mut self, p: u64) -> u64 {
        let Some((i, region)) = self.cell(p) else { return 0 };
        if !self.touched[i] {
            self.touched[i] = true;
            if region.shared != NONE && self.reads.len() < 1 << 14 {
                self.reads.push((shared_key(&self.m.shared[region.shared as usize], p as u32), self.mem[i] as u32));
            }
        }
        self.mem[i]
    }

    /// The effect of a write through `p` when it points into a groupshared global.
    fn shared_effect(&mut self, p: u64, fx: Fx, aux: u32, vals: [u64; 4]) {
        if let Some(r) = self.m.regions.get((p >> 32) as usize) {
            if r.shared != NONE {
                let target = fnv(&self.m.shared[r.shared as usize]);
                self.trace.push(Effect { fx, target, addr: [p as u32, 0, 0], aux, vals });
            }
        }
    }

    fn exec(&mut self, budget: u64, mut cov: Option<&mut Coverage>) -> R<(Stop, u64)> {
        let m = self.m;
        if let Some(c) = cov.as_deref_mut() {
            c.blocks[0] = true;
        }
        let mut pc = 0usize;
        let mut block = 0u32;
        let mut steps = 0u64;
        let mut phis: Vec<(u32, u64)> = Vec::new();
        loop {
            if steps >= budget {
                return Ok((Stop::Budget, steps));
            }
            steps += 1;
            let Some(&inst) = m.insts.get(pc) else { return Err("execution ran past the end of the function".into()) };
            pc += 1;
            let to = match inst {
                Inst::Bin { op, w, dst, a, b } => {
                    let v = int_bin(op, w, self.val(a), self.val(b));
                    self.set(dst, v);
                    continue;
                }
                Inst::FBin { op, w, dst, a, b } => {
                    let (a, b) = (self.val(a), self.val(b));
                    let v = fp_bin(op, w, a, b);
                    if let (Some(log), true) = (&mut self.nans, fp_is_nan(v, w)) {
                        log.push(format!("line {}: {op:?} {a:#x}, {b:#x} = {v:#x}", m.lines[pc - 1]));
                    }
                    self.set(dst, v);
                    continue;
                }
                Inst::ICmp { pred, w, dst, a, b } => {
                    let (a, b) = (self.val(a), self.val(b));
                    let v = int_cmp(pred, w, a, b);
                    if let Some(c) = cov.as_deref_mut() {
                        self.compared(c, pc, a, b, v);
                    }
                    self.set(dst, v as u64);
                    continue;
                }
                Inst::FCmp { pred, w, dst, a, b } => {
                    let (a, b) = (self.val(a), self.val(b));
                    let v = fp_cmp(pred, w, a, b);
                    if let Some(c) = cov.as_deref_mut() {
                        self.compared(c, pc, a, b, v);
                    }
                    self.set(dst, v as u64);
                    continue;
                }
                Inst::Cast { op, from, to, dst, a } => {
                    let v = cast(op, from, to, self.val(a));
                    self.set(dst, v);
                    continue;
                }
                Inst::Copy { dst, a } => {
                    let v = self.val(a);
                    self.set(dst, v);
                    continue;
                }
                Inst::Select { dst, c, a, b } => {
                    let v = if self.val(c) & 1 != 0 { self.val(a) } else { self.val(b) };
                    self.set(dst, v);
                    continue;
                }
                Inst::Phi { .. } => return Err("a phi after the head of its block".into()),
                Inst::Br { to } => to,
                Inst::CondBr { c, t, f } => {
                    if self.val(c) & 1 != 0 {
                        t
                    } else {
                        f
                    }
                }
                Inst::Switch { v, default, first, n } => {
                    let v = self.val(v);
                    let cases = &m.cases[first as usize..(first + n) as usize];
                    if let Some(cov) = cov.as_deref_mut() {
                        // the cases whose block no input has reached yet
                        for c in cases.iter().filter(|c| !cov.blocks[c.1 as usize]).take(8) {
                            if self.cmps.len() < 256 && (v | c.0) >> 32 == 0 {
                                self.cmps.push((0, v as u32, c.0 as u32));
                            }
                        }
                    }
                    cases.iter().find(|c| c.0 == v).map_or(default, |c| c.1)
                }
                Inst::Ret => return Ok((Stop::Returned, steps)),
                Inst::Unreachable => return Ok((Stop::Unreachable, steps)),
                Inst::DxOp { op, ovl, dst, first, n, agg, name } => {
                    self.dx_op(op, ovl, dst, &m.args[first as usize..(first + n) as usize], agg, name)?;
                    if self.nans.is_some() && ovl == OV_F32 && agg == NONE && dst != NONE && fp_is_nan(self.regs[dst as usize], 32) {
                        let args: Vec<String> = m.args[first as usize..(first + n) as usize].iter().map(|a| format!("{:#x}", self.val(*a))).collect();
                        let text = format!("line {}: {}({}) = {:#x}", m.lines[pc - 1], m.names[name as usize], args.join(", "), self.regs[dst as usize]);
                        self.nans.as_mut().expect("checked").push(text);
                    }
                    continue;
                }
                Inst::Unknown { name } => return Err(format!("call to @{}: no model for this function", m.names[name as usize])),
                Inst::ExtractValue { dst, a, idx } => {
                    let v = self.aggs.get(self.val(a) as usize).map_or(0, |x| x[idx as usize]);
                    self.set(dst, v);
                    continue;
                }
                Inst::Load { dst, p } => {
                    let p = self.val(p);
                    let v = self.load(p);
                    self.set(dst, v);
                    continue;
                }
                Inst::Store { v, p } => {
                    let (v, p) = (self.val(v), self.val(p));
                    if let Some((i, _)) = self.cell(p) {
                        self.mem[i] = v;
                        self.touched[i] = true;
                    }
                    self.shared_effect(p, Fx::SharedStore, 0, [v, 0, 0, 0]);
                    continue;
                }
                Inst::Gep { dst, base, off, first, n } => {
                    let p = self.val(base);
                    let mut o = (p as u32).wrapping_add(off as u32);
                    for &(idx, stride, w) in &m.gep[first as usize..(first + n) as usize] {
                        o = o.wrapping_add((sx(self.val(idx), w) as u32).wrapping_mul(stride as u32));
                    }
                    self.set(dst, (p & !0xFFFF_FFFF) | o as u64);
                    continue;
                }
                Inst::Alloca { dst, ptr } => {
                    self.set(dst, ptr);
                    continue;
                }
                Inst::AtomicRmw { op, w, dst, p, v } => {
                    let (p, v) = (self.val(p), self.val(v));
                    let old = self.load(p);
                    if let Some((i, _)) = self.cell(p) {
                        self.mem[i] = rmw(op, w, old, v);
                    }
                    self.shared_effect(p, Fx::SharedRmw, op as u32, [v, 0, 0, 0]);
                    self.set(dst, old);
                    continue;
                }
                Inst::CmpXchg { dst, agg, p, cmp, new } => {
                    let (p, cmp, new) = (self.val(p), self.val(cmp), self.val(new));
                    let old = self.load(p);
                    if old == cmp {
                        if let Some((i, _)) = self.cell(p) {
                            self.mem[i] = new;
                        }
                    }
                    self.shared_effect(p, Fx::SharedCas, 0, [cmp, new, 0, 0]);
                    self.aggs[agg as usize] = [old, (old == cmp) as u64, 0, 0, 0];
                    self.set(dst, agg as u64);
                    continue;
                }
            };
            // a branch: the phis of the target read their inputs before any of them is written
            let Some(&start) = m.blocks.get(to as usize) else { return Err("a branch to a block that does not exist".into()) };
            pc = start as usize;
            phis.clear();
            while let Some(&Inst::Phi { dst, first, n }) = m.insts.get(pc) {
                let Some(&(v, _)) = m.phi_in[first as usize..(first + n) as usize].iter().find(|i| i.1 == block) else {
                    return Err("a phi without an entry for the block control came from".into());
                };
                phis.push((dst, self.val(v)));
                pc += 1;
            }
            for &(dst, v) in &phis {
                self.set(dst, v);
            }
            steps += phis.len() as u64;
            if let Some(c) = cov.as_deref_mut() {
                c.visit(block, to);
            }
            block = to;
        }
    }
}

// ------------------------------------------------------------------------------------------
// dx.op calls, by opcode number (DxilConstants.h, OpCode).

fn ovl_fp(ovl: u8) -> R<u8> {
    match ovl {
        OV_F16 => Ok(16),
        OV_F32 => Ok(32),
        OV_F64 => Ok(64),
        _ => Err("a floating-point overload was expected".into()),
    }
}

fn ovl_int(ovl: u8) -> R<u8> {
    match ovl {
        1 | 8 | 16 | 32 | 64 => Ok(ovl),
        _ => Err("an integer overload was expected".into()),
    }
}

/// Width of a value of the overload type, as bits in a register.
fn ovl_bits(ovl: u8) -> u8 {
    match ovl {
        0 => 32,
        OV_F16 => 16,
        OV_F32 => 32,
        OV_F64 => 64,
        w => w,
    }
}

/// Unary float functions, in double and rounded to the overload: what DXC's constant folder
/// does (DxilConstantFolding.cpp), so a folded call and an executed one agree. Exp and Log
/// are base 2 in DXIL. Frc and Saturate follow the hardware where DXC's folder does not
/// (it folds frac(-x) to the positive fraction and saturate(NaN) to NaN).
fn fp_unary(op: u16, w: u8, a: u64) -> Option<u64> {
    let x = to_f64(a, w);
    let r = match op {
        6 => return Some(a & (mask(w) >> 1)),
        7 => return Some(from_f64(if x > 1.0 { 1.0 } else if x > 0.0 { x } else { 0.0 }, w)),
        12 => x.cos(),
        13 => x.sin(),
        14 => x.tan(),
        15 => x.acos(),
        16 => x.asin(),
        17 => x.atan(),
        18 => x.cosh(),
        19 => x.sinh(),
        20 => x.tanh(),
        21 => x.exp2(),
        22 => x - x.floor(),
        23 => x.log2(),
        24 => x.sqrt(),
        25 => 1.0 / x.sqrt(),
        26 => x.round_ties_even(),
        27 => x.floor(),
        28 => x.ceil(),
        29 => x.trunc(),
        _ => return None,
    };
    Some(nan_rule(w, &[a], from_f64(r, w)))
}

fn fp_mul_add(w: u8, a: u64, b: u64, c: u64) -> u64 {
    fp_bin(FOp::Add, w, fp_bin(FOp::Mul, w, a, b), c)
}

/// Bit-field extract as D3D defines it (ubfe / ibfe): width and offset use five bits.
fn bfe(width: u64, offset: u64, v: u64, signed: bool) -> u64 {
    let (width, offset, v) = ((width & 31) as u32, (offset & 31) as u32, v as u32);
    let shr = |x: u32, n: u32| if signed { ((x as i32) >> n) as u32 } else { x >> n };
    (if width == 0 {
        0
    } else if width + offset < 32 {
        shr(v << (32 - (width + offset)), 32 - width)
    } else {
        shr(v, offset)
    }) as u64
}

impl Run<'_> {
    /// The word a component of a resource element lives in. Raw buffers are byte addressed
    /// (so Load4 at 0 and Load at 4 meet), structured buffers by element and offset, typed
    /// buffers and textures by coordinates and component.
    fn key(h: u64, c: [u32; 3], comp: u32, word: u32, words: u32) -> Key {
        match handle_kind(h) {
            0 | KIND_RAW_BUFFER => (h, c[0].wrapping_add(c[1]).wrapping_add(4 * (comp * words + word)), 0, 0, 0),
            KIND_STRUCTURED_BUFFER => (h, c[0], c[1].wrapping_add(4 * (comp * words + word)), 0, 0),
            KIND_TYPED_BUFFER => (h, c[0], c[1], 0, 0x100 | comp | word << 12),
            _ => (h, c[0], c[1], c[2], 0x200 | comp | word << 12),
        }
    }

    fn word(&mut self, k: Key, float: bool) -> u32 {
        if let Some(&v) = self.res.get(&k) {
            return v;
        }
        let v = match self.over.get(&k) {
            Some(&v) => v,
            None => {
                let id = hash(&[k.0, k.1 as u64 | (k.2 as u64) << 32, k.3 as u64 | (k.4 as u64) << 32]);
                if float {
                    content_f32(self.seed, id)
                } else {
                    content(self.seed, id)
                }
            }
        };
        self.res.insert(k, v);
        if self.reads.len() < 1 << 14 {
            self.reads.push((k, v));
        }
        v
    }

    /// One component of an element: one word, or two for the 64-bit overloads.
    fn read(&mut self, h: u64, c: [u32; 3], comp: u32, ovl: u8) -> R<u64> {
        // float texels and typed-buffer elements read as plausible colours; raw memory as bits
        let float = ovl == OV_F32 && !matches!(handle_kind(h), 0 | KIND_RAW_BUFFER | KIND_STRUCTURED_BUFFER);
        match ovl_bits(ovl) {
            32 => Ok(self.word(Self::key(h, c, comp, 0, 1), float) as u64),
            64 => Ok(self.word(Self::key(h, c, comp, 0, 2), false) as u64 | (self.word(Self::key(h, c, comp, 1, 2), false) as u64) << 32),
            w => Err(format!("{w}-bit resource elements are not modelled")),
        }
    }

    fn write(&mut self, h: u64, c: [u32; 3], comp: u32, ovl: u8, v: u64) -> R<()> {
        match ovl_bits(ovl) {
            32 => {
                self.res.insert(Self::key(h, c, comp, 0, 1), v as u32);
            }
            64 => {
                self.res.insert(Self::key(h, c, comp, 0, 2), v as u32);
                self.res.insert(Self::key(h, c, comp, 1, 2), (v >> 32) as u32);
            }
            w => return Err(format!("{w}-bit resource elements are not modelled")),
        }
        Ok(())
    }

    /// A sample, gather or other read-only resource query: a function of everything asked.
    fn query(&self, op: u16, v: &[u64], comp: u64) -> u64 {
        let mut h = hash(&[self.seed, op as u64, comp]);
        for &x in v {
            h = mix(h ^ x);
        }
        h
    }

    fn dx_op(&mut self, op: u16, ovl: u8, dst: u32, args: &[Opd], agg: u32, name: u32) -> R<()> {
        let mut v = [0u64; 20];
        if args.len() > v.len() {
            return Err(format!("dx.op {} has more arguments than the interpreter reads", self.m.names[name as usize]));
        }
        for (k, a) in args.iter().enumerate() {
            v[k] = self.val(*a);
        }
        let v = &v[..args.len().max(12)];
        let c3 = |k: usize| [v[k] as u32, v[k + 1] as u32, v[k + 2] as u32];
        let unit = |h: u64| (((h >> 40) as f32) / 16_777_216.0).to_bits() as u64;
        let mut out = [0u64; 5];
        let result: u64 = match op {
            6 | 7 | 12..=29 => fp_unary(op, ovl_fp(ovl)?, v[1]).ok_or("unary float op")?,
            8..=11 => {
                let w = ovl_fp(ovl)?;
                let x = to_f64(v[1], w);
                let normal = match w {
                    16 => v[1] & 0x7C00 != 0,
                    32 => f32::from_bits(v[1] as u32).is_normal(),
                    _ => x.is_normal(),
                };
                (match op {
                    8 => x.is_nan(),
                    9 => x.is_infinite(),
                    10 => x.is_finite(),
                    _ => x.is_finite() && normal,
                }) as u64
            }
            30..=34 => {
                let w = ovl_int(ovl)?;
                let x = v[1] & mask(w);
                let lead = |x: u64| if x == 0 { u32::MAX } else { x.leading_zeros() - (64 - w as u32) };
                match op {
                    30 => x.reverse_bits() >> (64 - w),
                    31 => x.count_ones() as u64,
                    32 => (if x == 0 { u32::MAX } else { x.trailing_zeros() }) as u64,
                    33 => lead(x) as u64,
                    _ => lead(if sx(x, w) < 0 { !x & mask(w) } else { x }) as u64,
                }
            }
            // FMax / FMin: a NaN operand gives the other one; for equal operands (zeros of
            // either sign) the first, as LLVM's maxnum / minnum
            35 | 36 => {
                let w = ovl_fp(ovl)?;
                let (x, y) = (to_f64(v[1], w), to_f64(v[2], w));
                if x.is_nan() {
                    v[2]
                } else if y.is_nan() {
                    v[1]
                } else if (op == 35 && x < y) || (op == 36 && y < x) {
                    v[2]
                } else {
                    v[1]
                }
            }
            37..=40 => {
                let w = ovl_int(ovl)?;
                let (a, b) = (v[1] & mask(w), v[2] & mask(w));
                let (x, y) = (sx(a, w), sx(b, w));
                match op {
                    37 => x.max(y) as u64 & mask(w),
                    38 => x.min(y) as u64 & mask(w),
                    39 => a.max(b),
                    _ => a.min(b),
                }
            }
            41 | 42 => {
                let p = if op == 41 { (sx(v[1], 32) * sx(v[2], 32)) as u64 } else { (v[1] & mask(32)) * (v[2] & mask(32)) };
                out = [p >> 32, p & mask(32), 0, 0, 0];
                0
            }
            43 => {
                let (a, b) = (v[1] & mask(32), v[2] & mask(32));
                out = [int_bin(BinOp::UDiv, 32, a, b), int_bin(BinOp::URem, 32, a, b), 0, 0, 0];
                0
            }
            44 | 45 => {
                let (a, b) = (v[1] & mask(32), v[2] & mask(32));
                let r = if op == 44 { a + b } else { a.wrapping_sub(b) };
                out = [r & mask(32), (r >> 32) & 1, 0, 0, 0];
                0
            }
            46 => fp_mul_add(ovl_fp(ovl)?, v[1], v[2], v[3]),
            47 => {
                let w = ovl_fp(ovl)?;
                let r = to_f64(v[1], w).mul_add(to_f64(v[2], w), to_f64(v[3], w));
                nan_rule(w, &[v[1], v[2], v[3]], from_f64(r, w))
            }
            48 | 49 => v[1].wrapping_mul(v[2]).wrapping_add(v[3]) & mask(ovl_int(ovl)?),
            51 => bfe(v[1], v[2], v[3], true),
            52 => bfe(v[1], v[2], v[3], false),
            53 => {
                let (width, offset) = ((v[1] & 31) as u32, (v[2] & 31) as u32);
                let m = (((1u64 << width) - 1) << offset) & mask(32);
                ((v[3] << offset) & m) | (v[4] & !m & mask(32))
            }
            // Dot: the products added in order to a zero, as DXC folds it
            54..=56 => {
                let (w, n) = (ovl_fp(ovl)?, op as usize - 52);
                let mut sum = 0u64;
                for k in 0..n {
                    sum = fp_bin(FOp::Add, w, sum, fp_bin(FOp::Mul, w, v[1 + k], v[1 + n + k]));
                }
                sum
            }
            // CBufferLoadLegacy(handle, row)
            59 => {
                for k in 0..4 {
                    out[k as usize] = self.read(v[1], [v[2] as u32, 0, 0], k, if ovl_bits(ovl) == 64 { 64 } else { 32 })?;
                }
                0
            }
            // Sample .. SampleCmpLevelZero, TextureGather(Cmp), their later variants
            60..=65 | 73 | 74 | 223 | 224 | 254 | 255 => {
                for k in 0..4 {
                    out[k] = unit(self.query(op, v, k as u64));
                }
                0
            }
            // TextureLoad(srv, mip or sample, x, y, z, offsets)
            66 => {
                let c = [(v[3] as u32).wrapping_add(v[6] as u32), (v[4] as u32).wrapping_add(v[7] as u32), (v[5] as u32).wrapping_add(v[8] as u32)];
                for k in 0..4 {
                    out[k as usize] = if v[2] as u32 == 0 { self.read(v[1], c, k, ovl)? } else { self.query(op, &[v[1], v[2], c[0] as u64, c[1] as u64, c[2] as u64], k as u64) & mask(32) };
                }
                0
            }
            // TextureStore(uav, x, y, z, v0..v3, mask); BufferStore(uav, c0, c1, v0..v3, mask)
            67 | 69 => {
                let (c, first) = if op == 67 { (c3(2), 5) } else { ([v[2] as u32, v[3] as u32, 0], 4) };
                let wmask = v[first + 4] & 15;
                let mut vals = [0u64; 4];
                for k in 0..4 {
                    if wmask >> k & 1 != 0 {
                        vals[k] = v[first + k] & mask(ovl_bits(ovl));
                        self.write(v[1], c, k as u32, ovl, vals[k])?;
                    }
                }
                self.trace.push(Effect { fx: if op == 67 { Fx::TexStore } else { Fx::BufStore }, target: v[1], addr: c, aux: wmask as u32 | (ovl as u32) << 8, vals });
                0
            }
            // BufferLoad(srv, index, offset)
            68 => {
                for k in 0..4 {
                    out[k as usize] = self.read(v[1], [v[2] as u32, v[3] as u32, 0], k, ovl)?;
                }
                0
            }
            // BufferUpdateCounter(uav, +1 / -1): increment returns the old count, decrement the new
            70 => {
                let first = (self.query(op, &[v[1]], 0) % 1024) as u32;
                let n = self.counters.entry(v[1]).or_insert(first);
                let inc = v[2] as i8;
                let r = if inc >= 0 { *n } else { n.wrapping_sub(1) };
                *n = n.wrapping_add(inc as i32 as u32);
                self.trace.push(Effect { fx: Fx::Counter, target: v[1], addr: [0; 3], aux: 0, vals: [v[2] & 0xFF, 0, 0, 0] });
                r as u64
            }
            71 => 1,
            // GetDimensions(handle, mip)
            72 => {
                for k in 0..4 {
                    out[k] = self.query(op, &[v[1], v[2]], k as u64) % 4096 + 1;
                }
                0
            }
            // AtomicBinOp(handle, op, c0, c1, c2, value)
            78 => {
                let (h, c, w) = (v[1], c3(3), ovl_int(ovl)?);
                let old = self.read(h, c, 0, ovl)?;
                let new = match v[2] {
                    0 => rmw(RmwOp::Add, w, old, v[6]),
                    1 => rmw(RmwOp::And, w, old, v[6]),
                    2 => rmw(RmwOp::Or, w, old, v[6]),
                    3 => rmw(RmwOp::Xor, w, old, v[6]),
                    4 => rmw(RmwOp::Min, w, old, v[6]),
                    5 => rmw(RmwOp::Max, w, old, v[6]),
                    6 => rmw(RmwOp::UMin, w, old, v[6]),
                    7 => rmw(RmwOp::UMax, w, old, v[6]),
                    8 => rmw(RmwOp::Xchg, w, old, v[6]),
                    other => return Err(format!("atomicBinOp operation {other}")),
                };
                self.write(h, c, 0, ovl, new)?;
                self.trace.push(Effect { fx: Fx::Atomic, target: h, addr: c, aux: v[2] as u32 | (ovl as u32) << 8, vals: [v[6] & mask(w), 0, 0, 0] });
                old
            }
            // AtomicCompareExchange(handle, c0, c1, c2, compare, new)
            79 => {
                let (h, c, w) = (v[1], c3(2), ovl_int(ovl)?);
                let (cmp, new) = (v[5] & mask(w), v[6] & mask(w));
                let old = self.read(h, c, 0, ovl)?;
                if old == cmp {
                    self.write(h, c, 0, ovl, new)?;
                }
                self.trace.push(Effect { fx: Fx::AtomicCas, target: h, addr: c, aux: (ovl as u32) << 8, vals: [cmp, new, 0, 0] });
                old
            }
            80 => {
                self.trace.push(Effect { fx: Fx::Barrier, target: 0, addr: [0; 3], aux: v[1] as u32, vals: [0; 4] });
                0
            }
            81 => unit(self.query(op, v, 0)),
            // ThreadId, GroupId, ThreadIdInGroup (component), FlattenedThreadIdInGroup
            93 => match v[1] {
                0 => (self.at.group[0] * 64 + self.at.thread) as u64,
                k => self.at.group[(k as usize).min(2)] as u64,
            },
            94 => self.at.group[(v[1] as usize).min(2)] as u64,
            95 => {
                if v[1] == 0 {
                    self.at.thread as u64
                } else {
                    0
                }
            }
            96 => self.at.thread as u64,
            101 => (v[1] & mask(32)) | v[2] << 32,
            102 => {
                out = [v[1] & mask(32), v[1] >> 32, 0, 0, 0];
                0
            }
            // the wave is this lane alone, with LANE_COUNT lanes declared
            110 => 1,
            111 => self.at.lane as u64,
            112 => LANE_COUNT as u64,
            113 | 114 | 135 | 222 => v[1] & 1,
            115 => 1,
            116 | 165 => {
                let on = op == 165 || v[1] & 1 != 0;
                out = [(on as u64) << self.at.lane, 0, 0, 0, 0];
                0
            }
            // WaveReadLaneAt / QuadReadLaneAt: another lane holds some function of what this one holds
            117 | 122 => {
                let own = if op == 117 { self.at.lane } else { self.at.lane & 3 };
                if v[2] as u32 == own {
                    v[1]
                } else {
                    self.query(op, &[v[1], v[2] & mask(32)], 0) & mask(ovl_bits(ovl))
                }
            }
            118 | 119 | 120 => v[1],
            // WavePrefixOp, WaveMultiPrefixOp: nothing before this lane, so the identity
            121 | 166 => {
                let kind = if op == 121 { v[2] } else { v[6] };
                match (kind, ovl) {
                    (0, _) => 0,
                    (_, OV_F16 | OV_F32 | OV_F64) => from_f64(1.0, ovl_bits(ovl)),
                    _ => 1,
                }
            }
            123 => self.query(op, &[v[1], v[2]], 0) & mask(ovl_bits(ovl)),
            124..=129 => v[1],
            130 => f64_to_half(to_f64(v[1], 32)) as u64,
            131 => {
                let h = v[1] as u16;
                if fp_is_nan(h as u64, 16) {
                    ((h as u64 & 0x8000) << 16) | 0x7F80_0000 | ((h as u64 & 0x3FF) << 13)
                } else {
                    from_f64(half_to_f64(h), 32)
                }
            }
            132 => cast(CastOp::FpConv, 64, 32, v[1]),
            133 => cast(CastOp::FpToSi, 64, 32, v[1]),
            134 => cast(CastOp::FpToUi, 64, 32, v[1]),
            136 | 167 | 221 => 0,
            // RawBufferLoad(srv, index, elementOffset, mask, alignment)
            139 => {
                for k in 0..4 {
                    if v[4] >> k & 1 != 0 {
                        out[k as usize] = self.read(v[1], [v[2] as u32, v[3] as u32, 0], k, ovl)?;
                    }
                }
                0
            }
            // RawBufferStore(uav, index, elementOffset, v0..v3, mask, alignment)
            140 => {
                let c = [v[2] as u32, v[3] as u32, 0];
                let wmask = v[8] & 15;
                let mut vals = [0u64; 4];
                for k in 0..4 {
                    if wmask >> k & 1 != 0 {
                        vals[k] = v[4 + k] & mask(ovl_bits(ovl));
                        self.write(v[1], c, k as u32, ovl, vals[k])?;
                    }
                }
                self.trace.push(Effect { fx: Fx::RawStore, target: v[1], addr: c, aux: wmask as u32 | (ovl as u32) << 8, vals });
                0
            }
            // Dot4AddI8Packed / Dot4AddU8Packed(acc, a, b)
            163 | 164 => {
                let mut acc = v[1] as u32;
                for k in 0..4 {
                    let (a, b) = ((v[2] >> (8 * k)) as u8, (v[3] >> (8 * k)) as u8);
                    acc = acc.wrapping_add(if op == 163 { (a as i8 as i32 * b as i8 as i32) as u32 } else { a as u32 * b as u32 });
                }
                acc as u64
            }
            // AnnotateHandle(handle, { kind and flags, type }): the kind selects the addressing
            216 => {
                let props = self.aggs.get(v[2] as usize).copied().unwrap_or([0; 5]);
                (v[1] & !(0xFF << 48)) | (props[0] & 0xFF) << 48
            }
            // CreateHandleFromBinding({ lower, upper, space, class }, index, nonUniform)
            217 => {
                let bind = self.aggs.get(v[1] as usize).copied().unwrap_or([0; 5]);
                1 << 60 | (bind[3] & 0xF) << 56 | (bind[2] & 0xFFFF) << 32 | (v[2] & mask(32))
            }
            // CreateHandleFromHeap(index, samplerHeap, nonUniform)
            218 => 2 << 60 | (if v[2] & 1 != 0 { 3 } else { 0xF }) << 56 | (v[1] & mask(32)),
            // Unpack4x8(mode, packed): unsigned 0, signed 1
            219 => {
                for k in 0..4 {
                    let b = (v[2] >> (8 * k)) as u8;
                    out[k] = (if v[1] & 1 != 0 { b as i8 as i64 as u64 } else { b as u64 }) & mask(ovl_bits(ovl));
                }
                0
            }
            // Pack4x8(mode, x, y, z, w): truncate 0, clamp to u8 1, clamp to s8 2
            220 => {
                let w = ovl_bits(ovl);
                let mut r = 0u64;
                for k in 0..4 {
                    let x = sx(v[2 + k], w);
                    let b = match v[1] {
                        1 => x.clamp(0, 255) as u8,
                        2 => x.clamp(-128, 127) as i8 as u8,
                        _ => x as u8,
                    };
                    r |= (b as u64) << (8 * k);
                }
                r
            }
            _ => return Err(format!("dx.op {} (opcode {op}) is not modelled", self.m.names[name as usize])),
        };
        if agg != NONE {
            self.aggs[agg as usize] = out;
            self.set(dst, agg as u64);
        } else {
            self.set(dst, result);
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------
// The two compilers.

fn env_num(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn dxc_home() -> Option<PathBuf> {
    let home = std::env::var_os("XCL_DXIL_DXC").map(PathBuf::from).filter(|p| p.is_dir());
    if home.is_none() {
        eprintln!("lower_exec: XCL_DXIL_DXC (the dxc-linux folder) is not set: the comparison with DXC is skipped");
    }
    home
}

/// Where DXC's text is kept: XCL_EXEC_CACHE, else the work tree's scratch folder beside the
/// DXC folder when there is one, else the system's temporary folder.
fn cache_dir(dxc: &Path) -> PathBuf {
    if let Some(d) = std::env::var_os("XCL_EXEC_CACHE") {
        return d.into();
    }
    match dxc.parent().map(|p| p.join("scratch")) {
        Some(s) if s.is_dir() => s.join("lower-exec").join("dxc-cache"),
        _ => std::env::temp_dir().join("xcl-dxil-lower-exec"),
    }
}

/// What the dump's first line says: `// xdxc: -E <entry> -T <profile> <extra args>`.
struct Directive {
    entry: String,
    profile: String,
    extra: Vec<String>,
}

fn directive(src: &str) -> Directive {
    let mut d = Directive { entry: "main".into(), profile: "cs_6_6".into(), extra: Vec::new() };
    if let Some(rest) = src.lines().next().and_then(|l| l.trim().strip_prefix("// xdxc:")) {
        let mut words = rest.split_whitespace();
        while let Some(w) = words.next() {
            match w {
                "-E" => d.entry = words.next().unwrap_or("main").to_string(),
                "-T" => d.profile = words.next().unwrap_or("cs_6_6").to_string(),
                other => d.extra.push(other.to_string()),
            }
        }
    }
    d
}

/// DXC's disassembly of `src` at the options XCL uses, cached by file name and size.
fn dxc_text(dxc: &Path, src: &Path, d: &Directive) -> R<String> {
    static UNIQUE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let size = std::fs::metadata(src).map_err(|e| format!("{}: {e}", src.display()))?.len();
    let name = src.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let dir = cache_dir(dxc);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let out = dir.join(format!("{name}.{size}.ll"));
    if let Ok(text) = std::fs::read_to_string(&out) {
        if !text.is_empty() {
            return Ok(text);
        }
    }
    let exe = ["bin/dxc", "bin/dxc.exe", "dxc.exe", "dxc"].iter().map(|p| dxc.join(p)).find(|p| p.is_file()).ok_or_else(|| format!("no dxc under {}", dxc.display()))?;
    let mut lib = dxc.join("lib").into_os_string();
    if let Some(old) = std::env::var_os("LD_LIBRARY_PATH") {
        lib.push(":");
        lib.push(old);
    }
    let tmp = dir.join(format!("{name}.{size}.{}.{}.tmp", std::process::id(), UNIQUE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
    let run = std::process::Command::new(&exe).env("LD_LIBRARY_PATH", lib).args(["-E", &d.entry, "-T", &d.profile, "-HV", "2021", "-Gis", "-O1"]).args(&d.extra).arg("-Fc").arg(&tmp).arg(src).output().map_err(|e| format!("{}: {e}", exe.display()))?;
    if !run.status.success() {
        let _ = std::fs::remove_file(&tmp);
        let err = String::from_utf8_lossy(&run.stderr);
        return Err(err.lines().take(4).collect::<Vec<_>>().join(" | "));
    }
    std::fs::rename(&tmp, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    std::fs::read_to_string(&out).map_err(|e| format!("{}: {e}", out.display()))
}

enum OursError {
    Unsupported(String),
    Failed(String),
}

/// Our IR as text: parse, lower, print (what the bitcode writer would emit).
fn ours_text(src: &str, d: &Directive) -> Result<String, OursError> {
    let lowered = std::panic::catch_unwind(|| {
        let unit = xcl_dxil::hlsl::parse::parse(src)?;
        let (module, _) = xcl_dxil::hlsl::lower::lower(&unit, &d.entry, &d.profile)?;
        Ok::<_, xcl_dxil::Error>(xcl_dxil::irprint::module(&module))
    });
    match lowered {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(xcl_dxil::Error::Unsupported(what))) => Err(OursError::Unsupported(what)),
        Ok(Err(e)) => Err(OursError::Failed(e.to_string())),
        Err(panic) => {
            let msg = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            Err(OursError::Failed(format!("panic: {msg}")))
        }
    }
}

// ------------------------------------------------------------------------------------------
// The comparison.

struct Limits {
    seeds: u64,
    steps: u64,
    /// runs of the input search per program; 0 = by the size of the program
    search: u64,
    /// faults to inject into our IR to measure how sharp a pass is (XCL_EXEC_MUTANTS)
    mutants: u64,
}

impl Limits {
    fn from_env() -> Self {
        Limits { seeds: env_num("XCL_EXEC_SEEDS", 16).max(1), steps: env_num("XCL_EXEC_STEPS", 2_000_000), search: env_num("XCL_EXEC_SEARCH", 0), mutants: env_num("XCL_EXEC_MUTANTS", 0) }
    }
}

#[derive(Debug)]
enum Report {
    /// Every input agreed.
    Pass(Stats),
    Mismatch(String),
    OursUnsupported(String),
    OursFailed(String),
    /// The parser or the interpreter does not handle something (the text says in which IR).
    Interpreter(String),
    Dxc(String),
}

/// What a pass rests on: `inputs` inputs (the seeds and what the search added), of which
/// `budget` ran out of steps on a side (compared as far as the shorter trace); the traces
/// of DXC's IR had `fewest` to `most` effects and `distinct` different contents; the blocks
/// reached, of all blocks, in each IR.
#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    inputs: usize,
    budget: usize,
    fewest: usize,
    most: usize,
    distinct: usize,
    dxc_blocks: (usize, usize),
    our_blocks: (usize, usize),
    mutants: Mutants,
}

/// The first effect at which two runs differ. When a run stopped at the step budget its
/// trace is a prefix of what it would do, so it may be the shorter one; a run that finished
/// may not.
fn first_difference(a: &Outcome, b: &Outcome) -> Option<usize> {
    let n = a.trace.len().min(b.trace.len());
    if let Some(i) = (0..n).find(|&i| a.trace[i] != b.trace[i]) {
        return Some(i);
    }
    let shorter = if a.trace.len() < b.trace.len() { a } else { b };
    if a.trace.len() != b.trace.len() && shorter.stop != Stop::Budget {
        return Some(n);
    }
    // one finished and the other is still running after many times as many steps: it loops
    let hangs = |done: &Outcome, other: &Outcome| done.stop != Stop::Budget && other.stop == Stop::Budget && other.steps / 32 > done.steps + 1000;
    (hangs(a, b) || hangs(b, a)).then_some(n)
}

/// A resource word DXC's IR read (before anything was stored there) that ours did not.
/// DXC only ever removes reads (dead or repeated ones; a dx.op call is never speculated),
/// and we print only live instructions, so what DXC's IR reads our IR must read too. This
/// sees far more of a computation than the stores do: every address a program forms and
/// reads through, on every path. Groupshared cells are left out (LLVM may hoist a plain
/// load), and so is a run of ours that did not finish.
fn unread(dxc: &Outcome, ours: &Outcome) -> Option<(Key, u32)> {
    const FULL: usize = 1 << 14;
    if ours.stop == Stop::Budget || ours.reads.len() >= FULL || std::env::var_os("XCL_EXEC_NO_READS").is_some() {
        return None;
    }
    let read: std::collections::HashSet<Key> = ours.reads.iter().map(|r| r.0).collect();
    dxc.reads.iter().find(|r| r.0 .0 >> 60 != 3 && !read.contains(&r.0)).copied()
}

/// Whether some input tells the two programs apart.
fn told_apart(dxc: &Module, ours: &Module, inputs: &[Input], steps: u64) -> bool {
    inputs.iter().any(|input| match (run(dxc, input, steps, None), run(ours, input, steps, None)) {
        (Ok(a), Ok(b)) => first_difference(&a, &b).is_some() || unread(&a, &b).is_some(),
        _ => true,
    })
}

/// One instruction of our IR text changed into another that is still valid: the other
/// operation of a pair, the other predicate, swapped select arms or branch targets, a
/// constant one higher. `r` picks among the changes that apply.
fn mutate_line(line: &str, r: u64) -> Option<String> {
    let (head, rest) = match line.split_once(" = ") {
        Some((l, rest)) => (format!("{l} = "), rest),
        None => ("  ".to_string(), line.trim_start()),
    };
    let (op, tail) = rest.split_once(' ')?;
    fn other(pairs: &[(&'static str, &'static str)], w: &str) -> Option<&'static str> {
        pairs.iter().find_map(|&(a, b)| if w == a { Some(b) } else if w == b { Some(a) } else { None })
    }
    const OPS: [(&str, &str); 12] = [("add", "sub"), ("and", "or"), ("xor", "or"), ("shl", "lshr"), ("ashr", "lshr"), ("fadd", "fsub"), ("fmul", "fadd"), ("fdiv", "fmul"), ("zext", "sext"), ("mul", "add"), ("udiv", "urem"), ("sdiv", "udiv")];
    const PREDS: [(&str, &str); 9] = [("eq", "ne"), ("ult", "ule"), ("ugt", "uge"), ("slt", "sle"), ("sgt", "sge"), ("olt", "ole"), ("ogt", "oge"), ("oeq", "une"), ("ord", "uno")];
    match op {
        "icmp" | "fcmp" => {
            let (pred, operands) = tail.split_once(' ')?;
            Some(format!("{head}{op} {} {operands}", other(&PREDS, pred)?))
        }
        "select" | "br" => {
            let parts: Vec<&str> = tail.split(", ").collect();
            (parts.len() == 3 && parts[1] != parts[2]).then(|| format!("{head}{op} {}, {}, {}", parts[0], parts[2], parts[1]))
        }
        "call" => {
            let at = tail.find("@dx.op.binary.")?;
            let open = at + tail[at..].find("(i32 ")? + 5;
            let end = open + tail[open..].find(',')?;
            let opcode: u32 = tail[open..end].parse().ok()?;
            let swapped = if (35..=40).contains(&opcode) { ((opcode - 35) ^ 1) + 35 } else { return None };
            Some(format!("{head}call {}{swapped}{}", &tail[..open], &tail[end..]))
        }
        _ => {
            let swapped = other(&OPS, op)?;
            // sometimes the constant instead of the operation
            if r % 3 == 0 && !matches!(op, "zext" | "sext") {
                if let Some((before, last)) = tail.rsplit_once(", ") {
                    if let Ok(k) = last.parse::<i64>() {
                        return Some(format!("{head}{op} {before}, {}", k.wrapping_add(1)));
                    }
                }
            }
            Some(format!("{head}{swapped} {tail}"))
        }
    }
}

/// How sharp a pass is: `made` single-instruction faults were put into our IR, one at a
/// time; `reached` of them sit in a block the inputs run; `killed` were noticed.
#[derive(Debug, Default, Clone, Copy)]
struct Mutants {
    made: usize,
    reached: usize,
    killed: usize,
}

fn mutation_score(dxc: &Module, ours_text: &str, ours: &Module, inputs: &[Input], our_cov: &Coverage, limits: &Limits, verbose: bool) -> Mutants {
    let lines: Vec<&str> = ours_text.lines().collect();
    let body = lines.iter().position(|l| l.starts_with("define ")).unwrap_or(0);
    let candidates: Vec<usize> = (body..lines.len()).filter(|&i| lines[i].starts_with("  ") && mutate_line(lines[i], 1).is_some()).collect();
    let mut score = Mutants::default();
    if candidates.is_empty() {
        return score;
    }
    for k in 0..limits.mutants {
        let r = hash(&[0x4D55, k, ours_text.len() as u64]);
        let at = candidates[(r >> 8) as usize % candidates.len()];
        let Some(changed) = mutate_line(lines[at], r) else { continue };
        let mut text = String::with_capacity(ours_text.len() + 16);
        for (i, l) in lines.iter().enumerate() {
            text.push_str(if i == at { &changed } else { l });
            text.push('\n');
        }
        let Ok(mutant) = parse_module(&text) else { continue };
        score.made += 1;
        // the block of the changed instruction, by its line
        let inst = ours.lines.partition_point(|&l| (l as usize) < at + 1);
        let block = ours.blocks.partition_point(|&s| (s as usize) <= inst).saturating_sub(1);
        if !our_cov.blocks.get(block).copied().unwrap_or(false) {
            continue;
        }
        score.reached += 1;
        if told_apart(dxc, &mutant, inputs, limits.steps) {
            score.killed += 1;
        } else if verbose {
            println!("  a fault no input noticed, line {}: {}   was: {}", at + 1, changed.trim(), lines[at].trim());
        }
    }
    score
}

fn trace_hash(trace: &[Effect]) -> u64 {
    let mut h = trace.len() as u64;
    for e in trace {
        h = hash(&[h, e.fx as u64, e.target, e.addr[0] as u64, e.addr[1] as u64 | (e.addr[2] as u64) << 32, e.aux as u64, e.vals[0], e.vals[1], e.vals[2], e.vals[3]]);
    }
    h
}

/// Inputs that together reach as much of `m` as a bounded search finds: the plain seeds,
/// then inputs derived from kept ones by changing a few of the words they read (to the
/// constants the program compares against, to each other, by a bit) and kept when they take
/// a branch no earlier input took, or turn a comparison no earlier input turned. Random
/// content alone rarely gets past the first guards of a long program; this does, and it is
/// the same for both IRs since it only looks at one. Up to 256 further inputs are kept for
/// giving a trace not seen before: what a program computes shows only in what it stores, so
/// many different values through the stores are what makes a wrong instruction visible.
fn explore(m: &Module, limits: &Limits) -> R<(Vec<Input>, Coverage)> {
    let mut cov = Coverage::new(m);
    let mut kept: Vec<(Input, Vec<(Key, u32)>, Vec<(u32, u32, u32)>)> = Vec::new();
    let mut traces = std::collections::HashSet::new();
    let mut extra = Vec::new();
    for seed in 0..limits.seeds {
        let input = Input::seed(seed);
        let o = run(m, &input, limits.steps, Some(&mut cov))?;
        traces.insert(trace_hash(&o.trace));
        kept.push((input, o.reads, o.cmps));
    }
    let steps = limits.steps.min(200_000);
    // a long program has many guards in a row: 40 runs per block, within 600 to 50000
    let runs = if limits.search == 0 { (m.blocks.len() as u64 * 40).clamp(600, 50_000) } else { limits.search };
    let mut state = 0x5EED_0000 + m.insts.len() as u64;
    let mut next = move || {
        state = state.wrapping_add(1);
        mix(state)
    };
    for _ in 0..runs {
        // half of the time one of the newest inputs: they are the deepest
        let pick = if next() % 2 == 0 { kept.len() - 1 - (next() as usize % kept.len().min(8)) } else { next() as usize % kept.len() };
        let (parent, reads, cmps) = &kept[pick];
        if reads.is_empty() {
            continue;
        }
        let mut child = parent.clone();
        if next() % 16 == 0 {
            child.seed = next() % limits.seeds.max(8);
        }
        let lane = place(child.seed).lane;
        for _ in 0..1 + next() % 3 {
            // the words read last decide the branch the run ended at
            let at = if next() % 2 == 0 { reads.len() - 1 - (next() as usize % reads.len().min(16)) } else { next() as usize % reads.len() };
            let (mut key, mut old) = reads[at];
            let word = (next() >> 16) as u32;
            let choice = next() % 20;
            // a comparison that failed: move a word it may derive from by the difference,
            // at the bit position where the word seems to hold the compared value
            let mut moved = None;
            let open: Vec<(u32, u32)> = if choice >= 10 { cmps.iter().filter(|c| c.0 == 0 || cov.conds[c.0 as usize - 1] != 3).map(|c| (c.1, c.2)).collect() } else { Vec::new() };
            if !open.is_empty() {
                let (mut a, mut b) = open[next() as usize % open.len()];
                if next() % 3 == 0 {
                    (a, b) = (b, a);
                }
                let mut related = Vec::new();
                for (i, &(_, v)) in reads.iter().enumerate().rev().take(512) {
                    if v.wrapping_sub(a).wrapping_add(256) <= 512 {
                        related.push((i, 0));
                    } else if a >= 4 {
                        let field = u32::MAX >> a.leading_zeros();
                        related.extend((1..=a.leading_zeros()).filter(|s| (v >> s) & field == a).map(|s| (i, s)));
                    }
                }
                let delta = b.wrapping_sub(a).wrapping_add([0, 0, 0, 1, u32::MAX][next() as usize % 5]);
                let shift = if related.is_empty() {
                    [0, 0, 0, 4, 8, 12, 16, 20, 24, 28, word % 32][next() as usize % 11]
                } else {
                    let (i, s) = related[next() as usize % related.len()];
                    (key, old) = reads[i];
                    s
                };
                moved = Some(old.wrapping_add(delta << shift));
            }
            let from_dict = |r: u64| m.dict.get(r as usize % m.dict.len().max(1)).copied().unwrap_or(0);
            let new = match choice {
                _ if moved.is_some() => moved.unwrap_or(0),
                0 | 1 => from_dict(next()),
                2 => from_dict(next()).wrapping_add(1),
                3 => from_dict(next()).wrapping_sub(1),
                4 => old ^ 1 << (word % 32),
                5 => 0,
                6 => u32::MAX,
                7 => reads[word as usize % reads.len()].1,
                8 => old ^ 1 << lane,
                9 => old.wrapping_add(1),
                10 => 1 << lane,
                // a float of ordinary size: garbage bits make every result NaN or infinite,
                // and then a wrong instruction changes nothing that is stored
                11..=13 => (((word % 257) as f32 - 128.0) / 16.0).to_bits(),
                _ => content(word as u64, next()),
            };
            child.over.insert(key, new);
        }
        cov.fresh = 0;
        let o = run(m, &child, steps, Some(&mut cov))?;
        if cov.fresh > 0 {
            traces.insert(trace_hash(&o.trace));
            kept.push((child, o.reads, o.cmps));
        } else if extra.len() < 256 && o.stop != Stop::Budget && !o.trace.is_empty() && traces.insert(trace_hash(&o.trace)) {
            // compared, but not searched from: the frontier is where the search should stay
            extra.push(child);
        }
    }
    Ok((kept.into_iter().map(|k| k.0).chain(extra).collect(), cov))
}

fn compare(dxc: &Module, ours: &Module, ours_text: &str, limits: &Limits, verbose: bool) -> Report {
    let shared: HashMap<u64, String> = dxc.shared.iter().chain(&ours.shared).map(|n| (fnv(n), n.clone())).collect();
    let show = |o: &Outcome, i: usize| o.trace.get(i).map_or_else(|| format!("(no effect: {:?} after {} effects)", o.stop, o.trace.len()), |e| show_effect(e, &shared));
    let (inputs, dxc_cov) = match explore(dxc, limits) {
        Ok(found) => found,
        Err(e) => return Report::Interpreter(format!("DXC's IR: {e}")),
    };
    let mut our_cov = Coverage::new(ours);
    let mut stats = Stats { inputs: inputs.len(), fewest: usize::MAX, dxc_blocks: (dxc_cov.blocks_hit(), dxc.blocks.len()), ..Stats::default() };
    let mut contents = std::collections::HashSet::new();
    for (n, input) in inputs.iter().enumerate() {
        let a = match run(dxc, input, limits.steps, None) {
            Ok(o) => o,
            Err(e) => return Report::Interpreter(format!("DXC's IR, input {n}: {e}")),
        };
        let a_nans = NAN_LOG.with(|l| l.take());
        let b = match run(ours, input, limits.steps, Some(&mut our_cov)) {
            Ok(o) => o,
            Err(e) => return Report::Interpreter(format!("our IR, input {n}: {e}")),
        };
        if std::env::var_os("XCL_EXEC_NANS").is_some() && first_difference(&a, &b).is_some() {
            let b_nans = match run(ours, input, limits.steps, None) {
                Ok(_) => NAN_LOG.with(|l| l.take()),
                Err(_) => Vec::new(),
            };
            for (side, log) in [("DXC ", &a_nans), ("ours", &b_nans)] {
                for line in log.iter().take(100_000) {
                    println!("    NaN in {side}: {line}");
                }
            }
        }
        let what = format!("input {n} (seed {}, {} words chosen by the search, {:?})", input.seed, input.over.len(), place(input.seed));
        if verbose && n < limits.seeds as usize {
            println!("  {what}: DXC {:?} after {} steps, {} effects; ours {:?} after {} steps, {} effects", a.stop, a.steps, a.trace.len(), b.stop, b.steps, b.trace.len());
        }
        if verbose && std::env::var("XCL_EXEC_TRACE").is_ok_and(|v| v.parse() == Ok(n)) {
            for (side, o) in [("DXC ", &a), ("ours", &b)] {
                for (i, e) in o.trace.iter().enumerate().take(200) {
                    println!("    {side} {i:4}: {}", show_effect(e, &shared));
                }
            }
        }
        if verbose && std::env::var("XCL_EXEC_TRACE").is_ok_and(|v| v == "last") {
            if let Some(e) = a.trace.last() {
                println!("    input {n}: {} effects, the last: {}", a.trace.len(), show_effect(e, &shared));
            }
        }
        if let Some(i) = first_difference(&a, &b) {
            let mut text = format!("{what}, effect {i}:\n      DXC : {}\n      ours: {}", show(&a, i), show(&b, i));
            if i > 0 {
                text.push_str(&format!("\n      (the effect before, the same in both: {})", show(&a, i - 1)));
            }
            text.push_str(&format!("\n      DXC: {} effects, {:?} after {} steps; ours: {} effects, {:?} after {} steps", a.trace.len(), a.stop, a.steps, b.trace.len(), b.stop, b.steps));
            if !input.over.is_empty() && input.over.len() <= 12 {
                let mut words: Vec<_> = input.over.iter().collect();
                words.sort();
                text.push_str(&format!("\n      chosen words: {}", words.iter().map(|(k, v)| format!("{} [{:#x}, {:#x}] = {v:#x}", show_handle(k.0), k.1, k.2)).collect::<Vec<_>>().join("; ")));
            }
            return Report::Mismatch(text);
        }
        if let Some((k, v)) = unread(&a, &b) {
            return Report::Mismatch(format!(
                "{what}: the effects agree, but DXC's IR reads {} [{:#x}, {:#x}, {:#x}] (it held {v:#x}) and ours never does\n      DXC: {} words read, {} effects, {:?} after {} steps; ours: {} words read, {} effects, {:?} after {} steps",
                show_handle(k.0),
                k.1,
                k.2,
                k.3,
                a.reads.len(),
                a.trace.len(),
                a.stop,
                a.steps,
                b.reads.len(),
                b.trace.len(),
                b.stop,
                b.steps
            ));
        }
        stats.budget += (a.stop == Stop::Budget || b.stop == Stop::Budget) as usize;
        stats.fewest = stats.fewest.min(a.trace.len());
        stats.most = stats.most.max(a.trace.len());
        contents.insert(trace_hash(&a.trace));
    }
    stats.distinct = contents.len();
    stats.our_blocks = (our_cov.blocks_hit(), ours.blocks.len());
    if limits.mutants > 0 {
        stats.mutants = mutation_score(dxc, ours_text, ours, &inputs, &our_cov, limits, verbose);
    }
    if verbose && std::env::var_os("XCL_EXEC_UNREACHED").is_some() {
        for (side, m, cov) in [("DXC's IR", dxc, &dxc_cov), ("our IR", ours, &our_cov)] {
            let lines: Vec<String> = (0..m.blocks.len()).filter(|&b| !cov.blocks[b]).take(40).map(|b| m.block_lines[b].to_string()).collect();
            println!("  blocks of {side} no input reached start at lines {}", lines.join(" "));
            let frontier: Vec<String> = (0..m.blocks.len())
                .filter(|&b| {
                    let end = m.blocks.get(b + 1).map_or(m.insts.len(), |&e| e as usize);
                    cov.blocks[b]
                        && match m.insts[end - 1] {
                            Inst::CondBr { t, f, .. } => !cov.blocks[t as usize] || !cov.blocks[f as usize],
                            Inst::Switch { default, first, n, .. } => !cov.blocks[default as usize] || m.cases[first as usize..(first + n) as usize].iter().any(|c| !cov.blocks[c.1 as usize]),
                            _ => false,
                        }
                })
                .take(40)
                .map(|b| m.block_lines[b].to_string())
                .collect();
            println!("  reached blocks of {side} with a way out no input took start at lines {}", frontier.join(" "));
        }
    }
    Report::Pass(stats)
}

/// Compiles `path` both ways and compares. `reference` is DXC's text made earlier, for the
/// programs DXC takes minutes on.
fn check_file(dxc: &Path, path: &Path, reference: Option<&Path>, limits: &Limits, verbose: bool) -> Report {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return Report::Dxc(format!("{}: {e}", path.display())),
    };
    let d = directive(&src);
    let t = std::time::Instant::now();
    let ours = match ours_text(&src, &d) {
        Ok(text) => text,
        Err(OursError::Unsupported(what)) => return Report::OursUnsupported(what),
        Err(OursError::Failed(what)) => return Report::OursFailed(what),
    };
    let lowered = t.elapsed().as_secs_f64();
    if let Some(dir) = std::env::var_os("XCL_EXEC_DUMP") {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(Path::new(&dir).join(format!("{name}.ours.ll")), &ours);
    }
    let theirs = match reference {
        Some(r) => std::fs::read_to_string(r).map_err(|e| format!("{}: {e}", r.display())),
        None => dxc_text(dxc, path, &d),
    };
    let theirs = match theirs {
        Ok(text) => text,
        Err(e) => return Report::Dxc(e),
    };
    let t = std::time::Instant::now();
    let theirs = match parse_module(&theirs) {
        Ok(m) => m,
        Err(e) => return Report::Interpreter(format!("DXC's IR: {e}")),
    };
    let ours_text = ours;
    let ours = match parse_module(&ours_text) {
        Ok(m) => m,
        Err(e) => return Report::Interpreter(format!("our IR: {e}")),
    };
    if verbose {
        println!("  lowered and printed in {lowered:.3} s; both texts parsed in {:.3} s: DXC {} instructions in {} blocks, ours {} in {}", t.elapsed().as_secs_f64(), theirs.insts.len(), theirs.blocks.len(), ours.insts.len(), ours.blocks.len());
    }
    compare(&theirs, &ours, &ours_text, limits, verbose)
}

/// The lowering recurses on the syntax tree and the big programs are deep: every check runs
/// on a thread with room for that.
fn on_big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(512 << 20).spawn(f).expect("spawn").join().expect("the check thread panicked")
}

fn describe(report: &Report) -> String {
    match report {
        Report::Pass(s) => {
            let mut text = format!("passed: {} inputs, {} to {} effects each, {} different traces, {} stopped at the step budget; blocks reached: DXC {} of {}, ours {} of {}", s.inputs, s.fewest, s.most, s.distinct, s.budget, s.dxc_blocks.0, s.dxc_blocks.1, s.our_blocks.0, s.our_blocks.1);
            if s.mutants.made > 0 {
                text.push_str(&format!("; faults injected into our IR: {} made, {} in code the inputs run, {} noticed", s.mutants.made, s.mutants.reached, s.mutants.killed));
            }
            text
        }
        Report::Mismatch(text) => format!("MISMATCH at {text}"),
        Report::OursUnsupported(what) => format!("ours: unsupported: {what}"),
        Report::OursFailed(what) => format!("ours FAILED: {what}"),
        Report::Interpreter(what) => format!("interpreter: {what}"),
        Report::Dxc(what) => format!("dxc: {what}"),
    }
}

/// XCL_EXEC_SRC=<file> [XCL_EXEC_REF=<DXC's .ll>]: one program, with what every seed did.
#[test]
fn single() {
    let Some(src) = std::env::var_os("XCL_EXEC_SRC") else { return };
    let reference = std::env::var_os("XCL_EXEC_REF").map(PathBuf::from);
    let dxc = match (dxc_home(), &reference) {
        (Some(d), _) => d,
        (None, Some(_)) => PathBuf::new(),
        (None, None) => return,
    };
    let path = PathBuf::from(src);
    println!("{}:", path.display());
    let shown = path.clone();
    let report = on_big_stack(move || check_file(&dxc, &path, reference.as_deref(), &Limits::from_env(), true));
    println!("{}: {}", shown.display(), describe(&report));
    assert!(!matches!(report, Report::Mismatch(_)), "the traces differ");
}

/// XCL_EXEC_DIR=<dir>: every compute program there, as a table.
#[test]
fn corpus() {
    let Some(dir) = std::env::var_os("XCL_EXEC_DIR") else { return };
    let Some(dxc) = dxc_home() else { return };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir).expect("XCL_EXEC_DIR").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "hlsl")).collect();
    files.sort();
    files.retain(|p| {
        use std::io::BufRead;
        let mut first = String::new();
        std::fs::File::open(p).ok().map(std::io::BufReader::new).and_then(|mut f| f.read_line(&mut first).ok()).is_some() && first.contains("-T cs_6_6")
    });
    let limits = Limits::from_env();
    let jobs = env_num("XCL_EXEC_JOBS", 3).clamp(1, 3) as usize;
    println!("lower_exec: {} compute programs in {}: {} seeds each, {} steps per run, {jobs} at a time", files.len(), Path::new(&dir).display(), limits.seeds, limits.steps);
    let files = std::sync::Arc::new(files);
    let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let results = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let workers: Vec<_> = (0..jobs)
        .map(|_| {
            let (files, next, results, dxc) = (files.clone(), next.clone(), results.clone(), dxc.clone());
            std::thread::Builder::new()
                .stack_size(512 << 20)
                .spawn(move || {
                    let limits = Limits::from_env();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(path) = files.get(i) else { break };
                        let report = check_file(&dxc, path, None, &limits, false);
                        results.lock().expect("results").push((i, report));
                    }
                })
                .expect("spawn")
        })
        .collect();
    for w in workers {
        w.join().expect("a worker panicked");
    }
    let mut results = std::mem::take(&mut *results.lock().expect("results"));
    results.sort_by_key(|r| r.0);
    let name = |i: usize| files[i].file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();

    // the table: one line per class, then what is behind each class that is not a pass
    let mut classes: std::collections::BTreeMap<&str, Vec<(usize, String)>> = Default::default();
    let mut vacuous = Vec::new();
    let mut budgeted = Vec::new();
    let mut total = Stats::default();
    for (i, report) in &results {
        let (class, text) = match report {
            Report::Pass(s) => {
                if s.most == 0 {
                    vacuous.push(name(*i));
                }
                if s.budget > 0 {
                    budgeted.push(name(*i));
                }
                if std::env::var_os("XCL_EXEC_LIST").is_some() {
                    println!("{}: {}", name(*i), describe(report));
                }
                total.inputs += s.inputs;
                total.dxc_blocks = (total.dxc_blocks.0 + s.dxc_blocks.0, total.dxc_blocks.1 + s.dxc_blocks.1);
                total.our_blocks = (total.our_blocks.0 + s.our_blocks.0, total.our_blocks.1 + s.our_blocks.1);
                total.mutants = Mutants { made: total.mutants.made + s.mutants.made, reached: total.mutants.reached + s.mutants.reached, killed: total.mutants.killed + s.mutants.killed };
                ("1 passed", String::new())
            }
            Report::Mismatch(t) => ("2 MISMATCHED", t.clone()),
            Report::OursUnsupported(t) if t.starts_with("intrinsic") => ("4 ours unsupported: intrinsic", t.clone()),
            Report::OursUnsupported(t) => ("3 ours unsupported", t.clone()),
            Report::OursFailed(t) => ("5 ours failed", t.clone()),
            Report::Interpreter(t) => ("6 interpreter unsupported", t.clone()),
            Report::Dxc(t) => ("7 dxc failed", t.clone()),
        };
        classes.entry(class).or_default().push((*i, text));
    }
    println!();
    for class in ["1 passed", "2 MISMATCHED", "3 ours unsupported", "4 ours unsupported: intrinsic", "5 ours failed", "6 interpreter unsupported", "7 dxc failed"] {
        println!("{:32} {:4}", &class[2..], classes.get(class).map_or(0, |v| v.len()));
    }
    println!("{:32} {:4}", "total", results.len());
    println!("\nin the programs that passed: {} inputs compared; blocks reached: {} of {} in DXC's IR, {} of {} in ours", total.inputs, total.dxc_blocks.0, total.dxc_blocks.1, total.our_blocks.0, total.our_blocks.1);
    if total.mutants.made > 0 {
        println!("faults injected into our IR, one instruction each: {} made, {} in code the inputs run, {} noticed (a different trace, or a word DXC's IR reads left unread)", total.mutants.made, total.mutants.reached, total.mutants.killed);
    }
    println!("passed with no effect in any input (nothing was compared): {}{}", vacuous.len(), if vacuous.is_empty() { String::new() } else { format!("  e.g. {}", vacuous.iter().take(4).cloned().collect::<Vec<_>>().join(", ")) });
    println!("passed with an input stopped at the step budget (compared as a prefix): {}{}", budgeted.len(), if budgeted.is_empty() { String::new() } else { format!("  e.g. {}", budgeted.iter().take(4).cloned().collect::<Vec<_>>().join(", ")) });
    for (class, list) in &classes {
        if *class == "1 passed" {
            continue;
        }
        println!("\n{}:", &class[2..]);
        if *class == "2 MISMATCHED" {
            for (i, text) in list {
                println!("  {}: {text}", name(*i));
            }
            continue;
        }
        // grouped by message, with digits blanked so one cause is one group
        let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        for (i, text) in list {
            let key: String = text.split(": ").filter(|part| !part.starts_with("line ") && !part.starts_with("seed ") && !part.contains("IR, seed")).collect::<Vec<_>>().join(": ");
            groups.entry(key.chars().take(200).collect()).or_default().push(*i);
        }
        for (text, at) in groups {
            println!("  {:4}  {text}   e.g. {}", at.len(), at.iter().take(3).map(|i| name(*i)).collect::<Vec<_>>().join(", "));
        }
    }
    let mismatched = classes.get("2 MISMATCHED").map_or(0, |v| v.len());
    assert!(mismatched == 0, "{mismatched} programs behave differently from DXC's output");
}

// ------------------------------------------------------------------------------------------
// The harness tested on itself: hand-written IR with known answers (no DXC needed), then
// small HLSL programs through both compilers.

#[cfg(test)]
const TYPES: &str = "%dx.types.Handle = type { i8* }\n%dx.types.ResBind = type { i32, i32, i32, i8 }\n%dx.types.ResourceProperties = type { i32, i32 }\n%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }\n%dx.types.splitdouble = type { i32, i32 }\n";

#[cfg(test)]
fn run_text(body: &str, seed: u64, steps: u64) -> R<Outcome> {
    run(&parse_module(&format!("{TYPES}{body}"))?, &Input::seed(seed), steps, None)
}

#[test]
fn scalar_semantics() {
    assert_eq!(int_bin(BinOp::UDiv, 32, 5, 0), 0xFFFF_FFFF);
    assert_eq!(int_bin(BinOp::URem, 32, 5, 0), 5);
    assert_eq!(int_bin(BinOp::SDiv, 32, 0x8000_0000, 0xFFFF_FFFF), 0x8000_0000);
    assert_eq!(int_bin(BinOp::SDiv, 32, 0xFFFF_FFF9, 2), 0xFFFF_FFFD);
    assert_eq!(int_bin(BinOp::SRem, 32, 0xFFFF_FFF9, 2), 0xFFFF_FFFF);
    assert_eq!(int_bin(BinOp::AShr, 32, 0x8000_0000, 40), 0xFFFF_FFFF);
    assert_eq!(int_bin(BinOp::AShr, 32, 0x8000_0000, 4), 0xF800_0000);
    assert_eq!(int_bin(BinOp::Shl, 32, 1, 32), 0);
    assert_eq!(int_bin(BinOp::Mul, 64, u64::MAX, 3), u64::MAX - 2);
    assert_eq!(int_bin(BinOp::Add, 1, 1, 1), 0);
    assert!(int_cmp(IPred::Slt, 32, 0xFFFF_FFFF, 0) && !int_cmp(IPred::Ult, 32, 0xFFFF_FFFF, 0));
    assert_eq!(cast(CastOp::SExt, 8, 32, 0x80), 0xFFFF_FF80);
    assert_eq!(cast(CastOp::FpToUi, 32, 32, (-1.0f32).to_bits() as u64), 0);
    assert_eq!(cast(CastOp::FpToSi, 32, 32, (-1.5f32).to_bits() as u64), 0xFFFF_FFFF);
    assert_eq!(cast(CastOp::FpToSi, 32, 32, 3.0e10f32.to_bits() as u64), 0x7FFF_FFFF);
    assert_eq!(cast(CastOp::UiToFp, 32, 32, 0xFFFF_FFFF), 4294967296.0f32.to_bits() as u64);
    assert_eq!(cast(CastOp::SiToFp, 32, 32, 0xFFFF_FFFF), (-1.0f32).to_bits() as u64);
    assert_eq!(cast(CastOp::FpConv, 32, 64, 0.5f32.to_bits() as u64), 0.5f64.to_bits());
    let (zero, inf, nan) = (0f32.to_bits() as u64, f32::INFINITY.to_bits() as u64, 0x7FA0_0001u64);
    assert_eq!(fp_bin(FOp::Div, 32, zero, zero), 0x7FC0_0000);
    assert_eq!(fp_bin(FOp::Sub, 32, inf, inf), 0x7FC0_0000);
    assert_eq!(fp_bin(FOp::Add, 32, nan, inf), 0x7FE0_0001);
    assert_eq!(fp_bin(FOp::Mul, 32, nan, 0xFFFF_FFFF), fp_bin(FOp::Mul, 32, 0xFFFF_FFFF, nan));
    assert_eq!(fp_bin(FOp::Sub, 32, nan, 0xFFFF_FFFF), 0x7FE0_0001);
    assert!(fp_cmp(FPred::Une, 32, nan, nan) && !fp_cmp(FPred::Oeq, 32, nan, nan) && fp_cmp(FPred::Ult, 32, nan, zero));
    assert_eq!(fp_unary(22, 32, (-1.25f32).to_bits() as u64), Some(0.75f32.to_bits() as u64));
    assert_eq!(fp_unary(7, 32, nan), Some(0));
    assert_eq!(fp_unary(21, 32, 3f32.to_bits() as u64), Some(8f32.to_bits() as u64));
    assert_eq!(fp_unary(26, 32, 2.5f32.to_bits() as u64), Some(2f32.to_bits() as u64));
    assert_eq!(bfe(4, 4, 0xABCD, false), 0xC);
    assert_eq!(bfe(4, 4, 0xABCD, true), 0xFFFF_FFFC);
    assert_eq!(parse_fp("0x41F0000000000000", 32), Ok(4294967296.0f32.to_bits() as u64));
    assert_eq!(parse_fp("-1.500000e+00", 32), Ok((-1.5f32).to_bits() as u64));
    assert_eq!(parse_fp("0x7FF8000000000000", 32), Ok(0x7FC0_0000));
    assert_eq!(parse_fp("0xH3C00", 16), Ok(0x3C00));
    assert_eq!(parse_int("-2113396700").map(|v| v & mask(32)), Ok(2_181_570_596));
    assert_eq!(source_name("\\01?counter@@3IA"), "counter");
    assert_eq!(source_name("counter"), "counter");
}

#[test]
fn half_conversions() {
    for h in 0..=0xFFFFu16 {
        if !fp_is_nan(h as u64, 16) {
            assert_eq!(f64_to_half(half_to_f64(h)), h, "{h:#06x}");
        }
    }
    assert_eq!(f64_to_half(1.0), 0x3C00);
    assert_eq!(f64_to_half(65519.99), 0x7BFF);
    assert_eq!(f64_to_half(65520.0), 0x7C00);
    assert_eq!(f64_to_half(1.0 + 2f64.powi(-11)), 0x3C00);
    assert_eq!(f64_to_half(1.0 + 3.0 * 2f64.powi(-11)), 0x3C02);
    assert_eq!(f64_to_half(-2f64.powi(-25)), 0x8000);
    assert_eq!(f64_to_half(2f64.powi(-24) * 1.5), 0x0002);
}

#[cfg(test)]
const LOOP_IR: &str = r#"
@T = internal unnamed_addr constant [4 x i32] [i32 10, i32 20, i32 30, i32 40], align 4
@"\01?shared@@3PAIA" = external addrspace(3) global [2 x i32], align 4

define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 1, i32 1, i32 0, i8 1 }, i32 1, i1 false)  ; CreateHandleFromBinding(bind,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })  ; AnnotateHandle(res,props)  resource: RWByteAddressBuffer
  %3 = alloca [4 x i32], align 4
  br label %4

; <label>:4                                       ; preds = %4, %0
  %5 = phi i32 [ 0, %0 ], [ %9, %4 ]
  %6 = phi i32 [ 1, %0 ], [ %7, %4 ]
  %7 = phi i32 [ 2, %0 ], [ %6, %4 ]
  %8 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i32 0, i32 %5
  store i32 %6, i32* %8, align 4, !tbaa !16
  %9 = add nuw nsw i32 %5, 1
  %10 = icmp eq i32 %9, 4
  br i1 %10, label %11, label %4

; <label>:11                                      ; preds = %4
  %12 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i32 0, i32 2
  %13 = load i32, i32* %12, align 4, !tbaa !16
  %14 = load i32, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @T, i32 0, i32 2), align 4
  switch i32 %13, label %16 [
    i32 1, label %15
    i32 2, label %17
  ]

; <label>:15                                      ; preds = %11
  store i32 %14, i32 addrspace(3)* getelementptr inbounds ([2 x i32], [2 x i32] addrspace(3)* @"\01?shared@@3PAIA", i32 0, i32 1), align 4
  br label %17

; <label>:16                                      ; preds = %11
  unreachable

; <label>:17                                      ; preds = %15, %11
  %18 = phi i32 [ 100, %15 ], [ 200, %11 ]
  %19 = select i1 true, float 1.500000e+00, float 0x7FF8000000000000
  %20 = bitcast float %19 to i32
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 8, i32 undef, i32 %13, i32 %14, i32 %18, i32 %20, i8 15, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  ret void
}

declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #0
attributes #0 = { nounwind }
!16 = !{!17, !17, i64 0}
"#;

/// Phis read their inputs together (the two that swap give 1, 2, 1, 2), a switch, constant
/// and dynamic addresses, a groupshared store.
#[test]
fn interprets_control_flow_and_memory() {
    let o = run_text(LOOP_IR, 0, 1000).expect("run");
    assert_eq!(o.stop, Stop::Returned);
    let handle = 1u64 << 60 | 1 << 56 | KIND_RAW_BUFFER << 48 | 1;
    let expect = [
        Effect { fx: Fx::SharedStore, target: fnv("shared"), addr: [1, 0, 0], aux: 0, vals: [30, 0, 0, 0] },
        Effect { fx: Fx::RawStore, target: handle, addr: [8, 0, 0], aux: 15 | 32 << 8, vals: [1, 30, 100, 0x3FC0_0000] },
    ];
    assert_eq!(o.trace, expect);
    // the same program as we print it: plain names, no comments, other spacing
    let ours = LOOP_IR.replace("\"\\01?shared@@3PAIA\"", "shared").replace("                                      ; preds", "  ; preds").replace("  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)", ", !dx.precise !0");
    assert!(ours.contains("@shared = external") && ours.contains("; <label>:15  ; preds"));
    assert_eq!(run_text(&ours, 0, 1000).expect("run").trace, expect);
    // out of steps inside the loop: a prefix, and the comparison accepts it as one
    let short = run_text(LOOP_IR, 0, 12).expect("run");
    assert_eq!((short.stop, short.trace.len()), (Stop::Budget, 0));
    assert_eq!(first_difference(&short, &o), None);
    assert_eq!(first_difference(&o, &Outcome { trace: o.trace[..1].to_vec(), stop: Stop::Returned, steps: 0, reads: Vec::new(), cmps: Vec::new() }), Some(1));
}

#[test]
fn interprets_dx_ops() {
    let o = run_text(
        r#"
define void @main() {
  %1 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 7, i1 false, i1 false)
  %2 = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle %1, %dx.types.ResourceProperties { i32 4107, i32 0 })
  %3 = call i32 @dx.op.unaryBits.i32(i32 33, i32 1)
  %4 = call i32 @dx.op.unaryBits.i32(i32 32, i32 0)
  %5 = call i32 @dx.op.unaryBits.i32(i32 34, i32 -2)
  %6 = call float @dx.op.binary.f32(i32 35, float 0x7FF8000000000000, float 2.000000e+00), !dx.precise !0
  %7 = bitcast float %6 to i32
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 0, i32 undef, i32 %3, i32 %4, i32 %5, i32 %7, i8 15, i32 4)
  %8 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %2, i32 0, i32 undef, i8 15, i32 4)
  %9 = extractvalue %dx.types.ResRet.i32 %8, 2
  %10 = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle %2, i32 0, i32 8, i32 undef, i32 undef, i32 5)
  %11 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %2, i32 8, i32 undef, i8 1, i32 4)
  %12 = extractvalue %dx.types.ResRet.i32 %11, 0
  %13 = call double @dx.op.makeDouble.f64(i32 101, i32 0, i32 1073741824)
  %14 = fmul double %13, 1.500000e+00
  %15 = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double %14)
  %16 = extractvalue %dx.types.splitdouble %15, 1
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 16, i32 undef, i32 %9, i32 %10, i32 %12, i32 %16, i8 15, i32 4)
  ret void
}
"#,
        3,
        1000,
    )
    .expect("run");
    let vals: Vec<[u64; 4]> = o.trace.iter().map(|e| e.vals).collect();
    assert_eq!(vals, [[31, 0xFFFF_FFFF, 31, 0x4000_0000], [5, 0, 0, 0], [31, 31, 36, 0x4008_0000]]);
    assert_eq!((o.trace[1].fx, o.trace[1].addr), (Fx::Atomic, [8, 0, 0]));

    // unread memory is a function of the seed and the address only
    let load = |seed| {
        let text = "define void @main() {\n  %1 = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 7, i1 false, i1 false)\n  %2 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %1, i32 64, i32 undef, i8 3, i32 4)\n  %3 = extractvalue %dx.types.ResRet.i32 %2, 1\n  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 0, i32 undef, i32 %3, i32 undef, i32 undef, i32 undef, i8 1, i32 4)\n  ret void\n}\n";
        run_text(text, seed, 100).expect("run").trace[0].vals[0]
    };
    assert_eq!(load(5), load(5));
    assert_eq!(load(5), content(5, hash(&[2u64 << 60 | 0xF << 56 | 7, 68, 0])) as u64);
    assert!((0..16).map(load).collect::<std::collections::HashSet<_>>().len() > 8);

    // an operation without a model stops the run and is named
    let err = run_text("define void @main() {\n  call void @dx.op.discard(i32 82, i1 true)\n  ret void\n}\n", 0, 100).err().expect("unknown op");
    assert!(err.contains("dx.op.discard") && err.contains("82"), "{err}");
    let err = run_text("define void @main() {\n  call void @helper(i32 1)\n  ret void\n}\n", 0, 100).err().expect("unknown call");
    assert!(err.contains("@helper"), "{err}");
}

/// Small programs in the dialect, one construct family each. Every one reads its inputs
/// from `io`, so the seeds send it down different paths, and stores what it computed.
#[cfg(test)]
const SNIPPETS: &[(&str, &str)] = &[
    (
        "loops",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
uint scan(uint x, uint n, out uint steps) {
    uint acc = 0u;
    steps = 0u;
    for (uint i = 0u; i < 12u; i++) {
        steps++;
        if (i == n) break;
        if (((x >> i) & 1u) != 0u) continue;
        acc += i * 3u + (x & 0xFFu);
        if (acc > 600u) { acc ^= 0x5555u; break; }
    }
    return acc + 1u;
}
uint nest(uint a, uint b) {
    uint r = 0u;
    uint i = 0u;
    while (i < (a & 7u)) {
        uint j = 0u;
        do {
            if (j == (b & 3u)) { j++; continue; }
            r += (i * j) ^ a;
            if (r > 0x40000000u) break;
            j++;
        } while (j < 4u);
        i++;
    }
    return r | 0x80000000u;
}
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    uint b = io.Load(4u);
    uint s;
    res.Store(0u, scan(a, b & 15u, s));
    res.Store(4u, s);
    res.Store(8u, nest(a, b));
    if ((a & 1u) != 0u) { res.Store(12u, a); return; }
    res.Store(16u, b);
}
"#,
    ),
    (
        "return-in-loop",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
uint find(uint x, uint n) {
    uint acc = 0u;
    for (uint i = 0u; i < 12u; i++) {
        if (i == n) break;
        acc += i * 3u + (x & 0xFFu);
        if (acc > 600u) return acc ^ 0x5555u;
    }
    return acc + 1u;
}
uint nest(uint a, uint b) {
    uint r = 0u;
    for (uint i = 0u; i < (a & 7u); i++) {
        uint j = 0u;
        do {
            r += (i * j) ^ b;
            if ((r & 0x30u) == 0x30u) return r;
            j++;
        } while (j < 4u);
    }
    return r | 0x80000000u;
}
uint walk(uint x) {
    uint r = 0u;
    for (uint i = 0u; i < 6u; i++) {
        switch ((x >> (i * 2u)) & 3u) {
        case 0u: continue;
        case 1u: r += 1u; break;
        case 2u: r += 10u; if (r > 25u) return r; break;
        default: r += 100u; break;
        }
        r <<= 1;
    }
    return r;
}
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    uint b = io.Load(4u);
    res.Store(0u, find(a, b & 15u));
    res.Store(4u, nest(a, b));
    res.Store(8u, walk(io.Load(8u)));
}
"#,
    ),
    (
        "switch",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
uint pick(uint k, uint x) {
    switch (k) {
    case 0u: return x + 1u;
    case 1u:
    case 2u: x ^= 0xFFu; break;
    case 7u: { if (x > 100u) return 7u; x *= 3u; break; }
    default: x = x >> 1; break;
    case 9u: x = ~x; break;
    }
    return x + k;
}
uint walk(uint x) {
    uint r = 0u;
    for (uint i = 0u; i < 6u; i++) {
        switch ((x >> (i * 2u)) & 3u) {
        case 0u: continue;
        case 1u: r += 1u; break;
        case 2u: r += 10u; if (r > 25u) i = 6u; break;
        default: r += 100u; break;
        }
        r <<= 1;
    }
    return r;
}
[numthreads(1, 1, 1)]
void main() {
    for (uint i = 0u; i < 4u; i++) res.Store(i * 4u, pick(io.Load(i * 4u) % 11u, io.Load(64u + i * 4u)));
    res.Store(16u, walk(io.Load(32u)));
}
"#,
    ),
    (
        "out-inout",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
void split(uint v, out uint lo, out uint hi) { lo = v & 0xFFFFu; hi = v >> 16; }
void bump(inout uint v, uint by) { v += by; if (v > 1000u) v = 0u; }
bool take(inout uint budget, uint cost, out uint left) {
    left = budget;
    if (cost > budget) return false;
    budget -= cost;
    left = budget;
    return true;
}
void swap2(inout uint2 v) { v = v.yx; }
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    uint b = io.Load(4u);
    uint lo, hi;
    split(a, lo, hi);
    bump(lo, hi);
    bump(lo, b & 0xFFu);
    uint left;
    bool ok = take(hi, b & 0xFFFFu, left);
    uint2 p = uint2(lo, hi);
    if (ok) swap2(p);
    res.Store4(0u, uint4(p, left, ok ? 1u : 0u));
}
"#,
    ),
    (
        "vector-index",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint4 v = io.Load4(0u);
    uint i = io.Load(16u) & 3u;
    uint j = io.Load(20u) & 3u;
    v[i] = v[j] + 1u;
    v[(i + 1u) & 3u] ^= 0xFFu;
    float3 f = float3(1.5f, -2.0f, asfloat(io.Load(24u)));
    f[j % 3u] *= 2.0f;
    uint a[4] = { 1u, 2u, 3u, 4u };
    a[j] = v[i];
    res.Store4(0u, v);
    res.Store3(16u, asuint(f));
    res.Store(28u, a[i] + a[3u - i]);
}
"#,
    ),
    (
        "int64",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
void put(uint at, uint64_t v) { res.Store2(at, uint2(uint(v), uint(v >> 32))); }
[numthreads(1, 1, 1)]
void main() {
    uint64_t a = (uint64_t(io.Load(4u)) << 32) | uint64_t(io.Load(0u));
    uint64_t b = (uint64_t(io.Load(12u)) << 32) | uint64_t(io.Load(8u));
    uint n = io.Load(16u) & 63u;
    put(0u, a + b);
    put(8u, a * b);
    put(16u, (a << n) ^ (b >> n));
    put(24u, uint64_t(int64_t(a) >> n));
    put(32u, a / (b | 1u));
    put(40u, a % (b | 1u));
    res.Store(48u, (a < b ? 1u : 0u) | (int64_t(a) < int64_t(b) ? 2u : 0u) | (a == b ? 4u : 0u));
    put(56u, uint64_t(int64_t(int(io.Load(0u)))) - b);
}
"#,
    ),
    (
        "shift-divide",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    uint b = io.Load(4u);
    uint n = io.Load(8u);
    int sa = int(a);
    int sb = int(b);
    res.Store4(0u, uint4(a << n, a >> n, uint(sa >> n), a << (n & 7u)));
    res.Store4(16u, uint4(a / (b | 1u), a % (b | 1u), uint(sa / (sb | 1)), uint(sa % (sb | 1))));
    res.Store4(32u, uint4(a / 7u, uint(sa / -3), uint(sa % 16), a % 16u));
    res.Store4(48u, uint4(uint(sa / 8), uint(-sa), uint(abs(sa)), a * 0x01000193u));
    res.Store2(64u, uint2(uint(sa < sb), uint(a < b)));
}
"#,
    ),
    (
        "groupshared-atomics",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
groupshared uint g_counter;
groupshared uint g_table[8];
[numthreads(8, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint a = io.Load(0u);
    uint old;
    InterlockedAdd(g_counter, a & 0xFFu, old);
    InterlockedOr(g_table[gi & 7u], a);
    uint old2;
    InterlockedCompareExchange(g_table[(a >> 8) & 7u], 0u, a, old2);
    uint old3;
    InterlockedExchange(g_counter, 5u, old3);
    g_table[3] = old + old2;
    GroupMemoryBarrierWithGroupSync();
    res.Store(0u, g_table[a & 7u] + old3);
    if (gi == 0u) res.Store(4u, g_counter);
}
"#,
    ),
    (
        "resource-atomics",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    uint old;
    res.InterlockedAdd(8u, a, old);
    res.InterlockedXor(12u, old);
    uint seen;
    res.InterlockedCompareExchange(16u, a & 1u, 7u, seen);
    res.InterlockedCompareStore(20u, 0u, a);
    res.Store(24u, seen + res.Load(8u));
}
"#,
    ),
    (
        "float",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    float x = asfloat(io.Load(0u));
    float y = asfloat(io.Load(4u));
    uint k = io.Load(8u);
    res.Store4(0u, asuint(float4(x + y, x * y - 1.0f, min(x, y), max(x, y))));
    res.Store4(16u, uint4(asuint(saturate(x)), isnan(x) ? 1u : 0u, isinf(y) ? 1u : 0u, asuint(abs(x))));
    res.Store4(32u, uint4(f32tof16(x), asuint(f16tof32(k)), uint(x), uint(int(y))));
    res.Store4(48u, uint4(asuint(float(k)), asuint(float(int(k))), asuint(x < y ? x : y), x == y ? 1u : 0u));
    res.Store2(64u, asuint(float2(x / y, sqrt(abs(y)))));
}
"#,
    ),
    (
        "short-circuit",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
static uint calls = 0u;
bool probe(uint v, uint bit) { calls += 1u << (bit * 4u); return ((v >> bit) & 1u) != 0u; }
[numthreads(1, 1, 1)]
void main() {
    uint a = io.Load(0u);
    bool r1 = probe(a, 0u) && probe(a, 1u);
    bool r2 = probe(a, 2u) || probe(a, 3u);
    uint r3 = probe(a, 4u) ? (probe(a, 5u) ? 1u : 2u) : 3u;
    res.Store(0u, calls);
    res.Store(4u, (r1 ? 1u : 0u) | (r2 ? 2u : 0u) | (r3 << 4));
}
"#,
    ),
    (
        "tables",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
static const uint TABLE[8] = { 3u, 1u, 4u, 1u, 5u, 9u, 2u, 6u };
static const float WEIGHT[4] = { 0.25f, 0.5f, 1.0f, 2.0f };
[numthreads(1, 1, 1)]
void main() {
    uint i = io.Load(0u);
    uint acc[4] = { 0u, 0u, 0u, 0u };
    for (uint k = 0u; k < 8u; k++) acc[(TABLE[k] + i) & 3u] += k;
    res.Store4(0u, uint4(acc[0], acc[1], acc[2], acc[3]));
    res.Store2(16u, uint2(TABLE[(i >> 4) & 7u], asuint(WEIGHT[(i >> 8) & 3u] * float(TABLE[i & 7u]))));
}
"#,
    ),
    (
        "wave",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    uint lane = WaveGetLaneIndex();
    uint a = io.Load(0u);
    bool on = ((a >> lane) & 1u) != 0u;
    uint4 ballot = WaveActiveBallot(on);
    uint n = WaveActiveCountBits(on);
    uint first = WaveReadLaneFirst(a);
    uint other = WaveReadLaneAt(a + lane, (lane + 1u) & 31u);
    uint pre = WavePrefixCountBits(on);
    if (WaveIsFirstLane()) res.Store4(0u, ballot);
    res.Store4(16u, uint4(n, first, other, pre));
    res.Store2(32u, uint2(WaveGetLaneCount(), gi));
}
"#,
    ),
    (
        "vector-select",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint4 v = io.Load4(0u);
    uint4 w = io.Load4(16u);
    bool4 lt = v < w;
    res.Store4(0u, select(lt, v, w));
    res.Store4(16u, select(and(lt, v.wzyx > w), v + w, v - w));
    res.Store2(32u, (v.xy * w.zw).yx);
    res.Store(40u, select(or(lt.x, lt.y), 1u, 0u));
}
"#,
    ),
    (
        "vector-template",
        r#"
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[numthreads(1, 1, 1)]
void main() {
    uint off = io.Load(0u);
    uint lim = io.Load(4u);
    bool4 in0 = off + vector<uint, 4>(0u, 4u, 8u, 12u) < lim;
    res.Store4(0u, select(in0, uint4(1u, 2u, 3u, 4u), 0u));
    vector<float, 2> f = (vector<float, 2>)io.Load2(8u);
    res.Store2(16u, asuint(f * vector<float, 2>(0.5f, 2.0f)));
    res.Store4(24u, (vector<int, 4>(io.Load4(16u)) >> 1).wzyx);
}
"#,
    ),
];

/// With neither XCL_EXEC_DIR nor XCL_EXEC_SRC: the snippets through both compilers. A
/// snippet our lowering does not support yet is noted, not failed.
#[test]
fn builtin_snippets() {
    if std::env::var_os("XCL_EXEC_DIR").is_some() || std::env::var_os("XCL_EXEC_SRC").is_some() {
        return;
    }
    let Some(dxc) = dxc_home() else {
        // still prove that what we print parses and runs
        for (name, src) in SNIPPETS {
            match ours_text(src, &directive(src)) {
                Ok(text) => {
                    let m = parse_module(&text).unwrap_or_else(|e| panic!("{name}: our IR does not parse: {e}"));
                    let o = run(&m, &Input::seed(0), 100_000, None).unwrap_or_else(|e| panic!("{name}: our IR does not run: {e}"));
                    eprintln!("lower_exec: {name}: ours alone, {} effects", o.trace.len());
                }
                Err(OursError::Unsupported(what)) => eprintln!("lower_exec: {name}: ours: unsupported: {what}"),
                Err(OursError::Failed(what)) => panic!("{name}: ours failed: {what}"),
            }
        }
        return;
    };
    let dir = cache_dir(&dxc);
    std::fs::create_dir_all(&dir).expect("cache dir");
    let limits = Limits::from_env();
    let mut failed = Vec::new();
    for (name, src) in SNIPPETS {
        let path = dir.join(format!("builtin-{name}-{:016x}.hlsl", fnv(src)));
        std::fs::write(&path, src).expect("write snippet");
        let (d, p) = (dxc.clone(), path.clone());
        let l = Limits { ..limits };
        let report = on_big_stack(move || check_file(&d, &p, None, &l, false));
        println!("{name}: {}", describe(&report));
        match report {
            Report::Pass(s) => {
                if s.most == 0 || s.distinct < 2 {
                    failed.push(format!("{name}: the seeds do not exercise it"));
                }
            }
            Report::OursUnsupported(_) => {}
            other => failed.push(format!("{name}: {}", describe(&other))),
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
