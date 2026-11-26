//! Text → [`super::ast::Unit`] (owner: the parser subagent).
//!
//! A recursive-descent parser for the dialect `gpu::recomp::emit` produces, one token of
//! lookahead, the tree built directly into the unit's arenas. What is ambiguous in full HLSL
//! is decided by the dialect's own rules, stated in the module documentation of
//! [`super::ast`]: a type name is a built-in type word or a struct declared earlier, so
//! `(name)` before an expression is a cast exactly when `name` is one of those, and a
//! statement that starts with one is a declaration.
//!
//! HLSL outside the dialect is [`Error::Unsupported`] naming the construct, and text that is
//! not HLSL is [`Error::Source`]; neither panics, and nesting is bounded ([`MAX_DEPTH`]) so
//! that no input can exhaust the stack here or in the passes that walk the tree.

use std::collections::HashMap;

use super::ast::*;
use super::lex::{self, Lexer, Tok, Token, QUALS, TOK_COUNT};
use crate::Error;

/// Deepest nesting of statements and expressions in the text (a block, the body of an `if`
/// or a loop, an `else if`, a parenthesis, a call, an operand of a prefix operator or of
/// `?:` each count one): it bounds the parser's own recursion and the nesting of statements
/// in the tree. A 5 MB program of the title this backend exists for reaches 60.
pub const MAX_DEPTH: u32 = 512;

/// Tallest expression tree (nodes from an expression to its deepest leaf). Nesting in the
/// text is not what makes a tree tall: `a + b + c ...` is as deep as it is long. With
/// [`MAX_DEPTH`] this bounds the recursion of everything that walks the tree.
pub const MAX_HEIGHT: u32 = 1024;

/// The error is in the lexer's error slot ([`Lexer::set_error`]); this only unwinds.
struct Fail;
type R<T> = Result<T, Fail>;

const fn binary_table() -> [(u8, BinOp); TOK_COUNT] {
    let mut t = [(0u8, BinOp::Comma); TOK_COUNT];
    t[Tok::Star as usize] = (10, BinOp::Mul);
    t[Tok::Slash as usize] = (10, BinOp::Div);
    t[Tok::Percent as usize] = (10, BinOp::Rem);
    t[Tok::Plus as usize] = (9, BinOp::Add);
    t[Tok::Minus as usize] = (9, BinOp::Sub);
    t[Tok::Shl as usize] = (8, BinOp::Shl);
    t[Tok::Shr as usize] = (8, BinOp::Shr);
    t[Tok::Lt as usize] = (7, BinOp::Lt);
    t[Tok::Le as usize] = (7, BinOp::Le);
    t[Tok::Gt as usize] = (7, BinOp::Gt);
    t[Tok::Ge as usize] = (7, BinOp::Ge);
    t[Tok::EqEq as usize] = (6, BinOp::Eq);
    t[Tok::Ne as usize] = (6, BinOp::Ne);
    t[Tok::Amp as usize] = (5, BinOp::BitAnd);
    t[Tok::Caret as usize] = (4, BinOp::BitXor);
    t[Tok::Pipe as usize] = (3, BinOp::BitOr);
    t[Tok::AndAnd as usize] = (2, BinOp::LogAnd);
    t[Tok::OrOr as usize] = (1, BinOp::LogOr);
    t
}

/// Precedence (0: not a binary operator; higher binds tighter) and operator by token.
static BINARY: [(u8, BinOp); TOK_COUNT] = binary_table();

const fn continues_table() -> [bool; TOK_COUNT] {
    let binary = binary_table();
    let mut t = [false; TOK_COUNT];
    let mut i = 0;
    while i < TOK_COUNT {
        t[i] = binary[i].0 != 0;
        i += 1;
    }
    let more = [
        Tok::Question,
        Tok::Assign,
        Tok::PlusEq,
        Tok::MinusEq,
        Tok::StarEq,
        Tok::SlashEq,
        Tok::PercentEq,
        Tok::AmpEq,
        Tok::PipeEq,
        Tok::CaretEq,
        Tok::ShlEq,
        Tok::ShrEq,
    ];
    let mut i = 0;
    while i < more.len() {
        t[more[i] as usize] = true;
        i += 1;
    }
    t
}

/// Whether an assignment expression goes on with this token after a unary operand: a binary
/// operator, `?`, or an assignment operator.
static CONTINUES: [bool; TOK_COUNT] = continues_table();

struct Parser<'a> {
    lx: Lexer<'a>,
    u: Unit,
    /// lists being collected (arguments, block statements, switch cases): a list is copied
    /// to its pool in one piece when it ends, since lists nest
    expr_stack: Vec<ExprId>,
    stmt_stack: Vec<StmtId>,
    case_stack: Vec<Case>,
    depth: u32,
    /// height of the expression tree parsed last
    height: u32,
    /// 1 + the TypeId of each numeric type name used so far, by the token's `aux`
    num_types: Vec<u32>,
    type_ids: HashMap<TypeRef, TypeId>,
    /// struct names, behind a one-word filter (bit `sym % 64`): nearly every identifier is
    /// asked whether it is a type
    structs: HashMap<u32, u32>,
    struct_filter: u64,
    /// type and qualifier words that a declaration in scope uses as a name, innermost last:
    /// until its block ends the word is a plain identifier (DXC: `uint64_t half = ...`)
    shadows: Vec<u32>,
}

pub fn parse(src: &str) -> crate::Result<Unit> {
    if src.len() > i32::MAX as usize {
        return crate::unsupported("source larger than 2 GiB");
    }
    let mut p = Parser::new(src);
    p.bump();
    let r = p.unit();
    if let Some(e) = p.lx.take_error() {
        return Err(e);
    }
    match r {
        Ok(()) => Ok(p.u),
        Err(Fail) => Err(Error::Internal("the parser failed without an error".into())),
    }
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        let n = src.len();
        // measured on translated programs: an expression per 9 bytes of source, a statement
        // per 38, a declaration per 135. Not the names: a 50 MB program has 20 000, and the
        // lexer probes their table for every word, so it starts small and doubles as they need
        let mut u = Unit { syms: Interner::with_capacity(4096), ..Unit::default() };
        u.exprs.reserve(n / 7 + 64);
        u.stmts.reserve(n / 30 + 16);
        u.stmt_lines.reserve(n / 30 + 16);
        u.expr_lists.reserve(n / 24 + 16);
        u.stmt_lists.reserve(n / 30 + 16);
        u.vars.reserve(n / 100 + 16);
        let lx = Lexer::new(src, &mut u.syms);
        Parser {
            lx,
            u,
            expr_stack: Vec::with_capacity(64),
            stmt_stack: Vec::with_capacity(256),
            case_stack: Vec::new(),
            depth: 0,
            height: 0,
            num_types: vec![0; lex::NUM_TYPES],
            type_ids: HashMap::new(),
            structs: HashMap::new(),
            struct_filter: 0,
            shadows: Vec::new(),
        }
    }

    #[inline]
    fn bump(&mut self) {
        self.lx.advance(&mut self.u.syms);
        if !self.shadows.is_empty() {
            self.shadow();
        }
    }

    /// Makes the current token a plain identifier if a declaration in scope took its word.
    #[cold]
    fn shadow(&mut self) {
        let t = self.lx.tok;
        if lex::is_name(t.kind) && (t.kind != Tok::Ident || t.aux != 0) && self.shadows.contains(&(t.val as u32)) {
            self.lx.tok = Token { kind: Tok::Ident, aux: 0, ..t };
        }
    }

    #[cold]
    fn source(&mut self, msg: String) -> Fail {
        self.lx.set_error(Error::Source { line: self.lx.tok.line, msg });
        Fail
    }

    #[cold]
    fn unsupported(&mut self, what: String) -> Fail {
        self.lx.set_error(Error::Unsupported(what));
        Fail
    }

    /// The current token, for an error message.
    fn found(&self) -> String {
        let t = self.lx.tok;
        let text = match t.kind {
            Tok::Eof | Tok::Err => return "the end of the source".into(),
            Tok::Int | Tok::Float => return "a number".into(),
            Tok::Str => return "a string".into(),
            Tok::MacroParam => return "a macro parameter".into(),
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Semi => ";",
            Tok::Comma => ",",
            Tok::Dot => ".",
            Tok::Colon => ":",
            Tok::Question => "?",
            Tok::Tilde => "~",
            Tok::Bang => "!",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::Percent => "%",
            Tok::Amp => "&",
            Tok::Pipe => "|",
            Tok::Caret => "^",
            Tok::Lt => "<",
            Tok::Gt => ">",
            Tok::Shl => "<<",
            Tok::Shr => ">>",
            Tok::Le => "<=",
            Tok::Ge => ">=",
            Tok::EqEq => "==",
            Tok::Ne => "!=",
            Tok::AndAnd => "&&",
            Tok::OrOr => "||",
            Tok::Assign => "=",
            Tok::PlusEq => "+=",
            Tok::MinusEq => "-=",
            Tok::StarEq => "*=",
            Tok::SlashEq => "/=",
            Tok::PercentEq => "%=",
            Tok::AmpEq => "&=",
            Tok::PipeEq => "|=",
            Tok::CaretEq => "^=",
            Tok::ShlEq => "<<=",
            Tok::ShrEq => ">>=",
            Tok::PlusPlus => "++",
            Tok::MinusMinus => "--",
            Tok::ColonColon => "::",
            Tok::Arrow => "->",
            Tok::Hash => "#",
            Tok::HashHash => "##",
            _ => self.u.syms.name(Sym(t.val as u32)),
        };
        format!("'{text}'")
    }

    /// The error for a token that cannot be where it is: HLSL the dialect lacks if the token
    /// itself says so, else a syntax error.
    #[cold]
    fn unexpected(&mut self, wanted: &str) -> Fail {
        match self.lx.tok.kind {
            Tok::KwUnsupported => {
                let name = self.u.syms.name(Sym(self.lx.tok.val as u32));
                let class = if name.starts_with(|c: char| c.is_ascii_digit()) || name.contains(|c: char| c.is_ascii_digit()) { "type" } else { "keyword" };
                let what = format!("{class} {name}");
                self.unsupported(what)
            }
            Tok::ColonColon => self.unsupported("scope operator ::".into()),
            Tok::Arrow => self.unsupported("operator ->".into()),
            Tok::KwPackoffset => self.unsupported("packoffset".into()),
            _ => {
                let msg = format!("expected {wanted}, found {}", self.found());
                self.source(msg)
            }
        }
    }

    #[inline]
    fn expect(&mut self, kind: Tok, wanted: &'static str) -> R<()> {
        if self.lx.tok.kind == kind {
            self.bump();
            Ok(())
        } else {
            Err(self.unexpected(wanted))
        }
    }

    #[inline]
    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.unsupported(format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        Ok(())
    }

    /// Records the height of the expression just built.
    #[inline]
    fn grow(&mut self, height: u32) -> R<()> {
        self.height = height;
        if height > MAX_HEIGHT {
            return Err(self.unsupported(format!("expression tree deeper than {MAX_HEIGHT} nodes")));
        }
        Ok(())
    }

    #[inline]
    fn push(&mut self, e: Expr) -> ExprId {
        let id = ExprId(self.u.exprs.len() as u32);
        self.u.exprs.push(e);
        id
    }

    #[inline]
    fn push_stmt(&mut self, s: Stmt, line: u32) -> StmtId {
        let id = StmtId(self.u.stmts.len() as u32);
        self.u.stmts.push(s);
        self.u.stmt_lines.push(line);
        id
    }

    fn intern_type(&mut self, t: TypeRef) -> TypeId {
        if let Some(&id) = self.type_ids.get(&t) {
            return id;
        }
        let id = TypeId(self.u.types.len() as u32);
        self.u.types.push(t.clone());
        self.type_ids.insert(t, id);
        id
    }

    #[inline]
    fn num_type(&mut self, aux: u16) -> TypeId {
        let known = self.num_types[aux as usize];
        if known != 0 {
            return TypeId(known - 1);
        }
        let (kind, rows, cols) = lex::num_type(aux);
        let id = self.intern_type(TypeRef::Num { kind, rows, cols });
        self.num_types[aux as usize] = id.0 + 1;
        id
    }

    #[inline]
    fn is_struct(&self, sym: u64) -> bool {
        self.struct_filter >> (sym & 63) & 1 != 0 && self.structs.contains_key(&(sym as u32))
    }

    /// Whether the current token starts a declaration where a statement may also start.
    #[inline]
    fn at_local_decl(&self) -> bool {
        match self.lx.tok.kind {
            Tok::Ident => self.lx.tok.aux != 0 || self.is_struct(self.lx.tok.val),
            Tok::TyNum | Tok::TyRes | Tok::TySampler | Tok::TyStream | Tok::TyPatch | Tok::KwVector | Tok::KwMatrix => true,
            _ => false,
        }
    }

    // ---- top level ----

    fn unit(&mut self) -> R<()> {
        loop {
            match self.lx.tok.kind {
                Tok::Eof => return Ok(()),
                Tok::Semi => self.bump(),
                Tok::KwStruct => self.struct_decl()?,
                _ => self.global()?,
            }
        }
    }

    fn struct_decl(&mut self) -> R<()> {
        self.bump();
        if self.lx.tok.kind == Tok::LBrace {
            return Err(self.unsupported("struct without a name".into()));
        }
        let name = self.decl_name()?;
        match self.lx.tok.kind {
            Tok::LBrace => self.bump(),
            Tok::Colon => return Err(self.unsupported("struct inheritance".into())),
            Tok::Semi => return Err(self.unsupported("struct declared without a body".into())),
            _ => return Err(self.unexpected("'{'")),
        }
        let mut fields = Vec::new();
        while self.lx.tok.kind != Tok::RBrace {
            if self.lx.tok.kind == Tok::KwStruct {
                return Err(self.unsupported("struct declared inside a struct".into()));
            }
            let quals = self.quals();
            let ty = self.type_spec()?;
            loop {
                let line = self.lx.tok.line;
                let name = self.field_name()?;
                if self.lx.tok.kind == Tok::LParen {
                    return Err(self.unsupported("member function of a struct".into()));
                }
                let dims = self.dims()?;
                let (semantic, register) = self.bindings()?;
                if self.lx.tok.kind == Tok::Assign {
                    return Err(self.unsupported("default value of a struct field".into()));
                }
                fields.push(VarDecl { name, ty, dims, init: None, quals, semantic, register, line });
                if self.lx.tok.kind != Tok::Comma {
                    break;
                }
                self.bump();
            }
            self.expect(Tok::Semi, "';'")?;
        }
        self.bump();
        let index = self.u.structs.len() as u32;
        self.u.structs.push(Struct { name, fields });
        self.u.globals.push(Global::Struct(index));
        self.structs.insert(name.0, index);
        self.struct_filter |= 1 << (name.0 & 63);
        if self.lx.tok.kind != Tok::Semi {
            // `struct S { ... } a, b;`
            let ty = self.intern_type(TypeRef::Named(name));
            let first = self.u.vars.len() as u32;
            self.declarators(0, ty)?;
            for i in first..self.u.vars.len() as u32 {
                self.u.globals.push(Global::Var(i));
            }
            return Ok(());
        }
        self.bump();
        Ok(())
    }

    fn global(&mut self) -> R<()> {
        let attrs = if self.lx.tok.kind == Tok::LBracket { self.attrs()? } else { Vec::new() };
        let line = self.lx.tok.line;
        let quals = self.quals();
        let ty = self.type_spec()?;
        let name_line = self.lx.tok.line;
        let name = self.decl_name()?;
        if self.lx.tok.kind == Tok::LParen {
            return self.function(attrs, quals, ty, name, line);
        }
        if !attrs.is_empty() {
            return Err(self.source("attribute before a declaration that is not a function".into()));
        }
        let first = self.u.vars.len() as u32;
        self.declarators_from(quals, ty, name, name_line)?;
        for i in first..self.u.vars.len() as u32 {
            self.u.globals.push(Global::Var(i));
        }
        Ok(())
    }

    fn function(&mut self, attrs: Vec<Attr>, quals: u32, ret: TypeId, name: Sym, line: u32) -> R<()> {
        if quals & !(qual::STATIC | qual::INLINE) != 0 {
            let what = format!("qualified function {}", self.u.syms.name(name));
            return Err(self.unsupported(what));
        }
        self.bump();
        let scope = self.shadows.len();
        let mut params = Vec::new();
        if self.lx.tok.kind == Tok::KwVoid {
            // `f(void)`
            self.bump();
            if self.lx.tok.kind != Tok::RParen {
                return Err(self.unexpected("')'"));
            }
        }
        while self.lx.tok.kind != Tok::RParen {
            let line = self.lx.tok.line;
            let quals = self.quals();
            let ty = self.type_spec()?;
            if matches!(self.lx.tok.kind, Tok::Comma | Tok::RParen) {
                return Err(self.unsupported("parameter without a name".into()));
            }
            let name = self.decl_name()?;
            let dims = self.dims()?;
            let (semantic, register) = self.bindings()?;
            if self.lx.tok.kind == Tok::Assign {
                return Err(self.unsupported("default value of a parameter".into()));
            }
            params.push(VarDecl { name, ty, dims, init: None, quals, semantic, register, line });
            match self.lx.tok.kind {
                Tok::Comma => {
                    self.bump();
                    if self.lx.tok.kind == Tok::RParen {
                        return Err(self.unexpected("a parameter"));
                    }
                }
                Tok::RParen => {}
                _ => return Err(self.unexpected("',' or ')'")),
            }
        }
        self.bump();
        let mut ret_semantic = None;
        if self.lx.tok.kind == Tok::Colon {
            self.bump();
            if self.lx.tok.kind != Tok::Ident {
                return Err(self.unexpected("a semantic"));
            }
            ret_semantic = Some(Sym(self.lx.tok.val as u32));
            self.bump();
        }
        let body = match self.lx.tok.kind {
            Tok::Semi => {
                self.shadows.truncate(scope);
                self.bump();
                None
            }
            Tok::LBrace => {
                self.bump();
                Some(self.block_body(scope)?)
            }
            _ => return Err(self.unexpected("'{' or ';'")),
        };
        let index = self.u.funcs.len() as u32;
        self.u.funcs.push(Func { name, ret, ret_semantic, params, body, attrs, line });
        self.u.globals.push(Global::Func(index));
        Ok(())
    }

    // ---- declarations ----

    /// The qualifier words at the current position, as [`qual`] bits.
    fn quals(&mut self) -> u32 {
        let mut q = 0;
        while self.lx.tok.kind == Tok::Ident && self.lx.tok.aux != 0 {
            q |= QUALS[self.lx.tok.aux as usize - 1].1;
            self.bump();
        }
        q
    }

    /// The name a declaration declares. A type or qualifier word that DXC does not reserve
    /// is a name from here to the end of the scope.
    fn decl_name(&mut self) -> R<Sym> {
        let t = self.lx.tok;
        if !Lexer::can_be_declared(t) {
            return Err(self.unexpected("a name"));
        }
        if t.kind != Tok::Ident || t.aux != 0 {
            self.shadows.push(t.val as u32);
        }
        self.bump();
        Ok(Sym(t.val as u32))
    }

    /// The name of a struct field: any word that is not reserved; it hides nothing.
    fn field_name(&mut self) -> R<Sym> {
        let t = self.lx.tok;
        if !Lexer::can_be_declared(t) {
            return Err(self.unexpected("a name"));
        }
        self.bump();
        Ok(Sym(t.val as u32))
    }

    fn type_spec(&mut self) -> R<TypeId> {
        let t = self.lx.tok;
        match t.kind {
            Tok::TyNum => {
                self.bump();
                Ok(self.num_type(t.aux))
            }
            Tok::KwVoid => {
                self.bump();
                Ok(self.intern_type(TypeRef::Void))
            }
            Tok::Ident if self.is_struct(t.val) => {
                self.bump();
                Ok(self.intern_type(TypeRef::Named(Sym(t.val as u32))))
            }
            Tok::Ident => {
                let what = format!("type {}", self.u.syms.name(Sym(t.val as u32)));
                Err(self.unsupported(what))
            }
            Tok::TyRes => {
                self.bump();
                let kind = lex::RES_KINDS[t.aux as usize].1;
                let (mut elem, mut samples) = (None, None);
                if self.lx.tok.kind == Tok::Lt {
                    self.bump();
                    elem = Some(self.type_spec()?);
                    if self.lx.tok.kind == Tok::Comma {
                        self.bump();
                        samples = Some(self.template_int()?);
                    }
                    self.close_template()?;
                }
                Ok(self.intern_type(TypeRef::Resource { kind, elem, samples }))
            }
            Tok::TySampler => {
                self.bump();
                Ok(self.intern_type(TypeRef::Sampler { comparison: t.aux != 0 }))
            }
            Tok::TyStream => {
                self.bump();
                self.expect(Tok::Lt, "'<'")?;
                let elem = self.type_spec()?;
                self.close_template()?;
                let kind = [StreamKind::Point, StreamKind::Line, StreamKind::Triangle][t.aux as usize % 3];
                Ok(self.intern_type(TypeRef::Stream { kind, elem }))
            }
            Tok::TyPatch => {
                self.bump();
                self.expect(Tok::Lt, "'<'")?;
                let elem = self.type_spec()?;
                self.expect(Tok::Comma, "','")?;
                let points = self.template_int()?;
                self.close_template()?;
                Ok(self.intern_type(TypeRef::Patch { output: t.aux != 0, elem, points }))
            }
            Tok::KwVector | Tok::KwMatrix => {
                // `vector<T, n>`, `matrix<T, r, c>`; bare: float4 and float4x4
                self.bump();
                let matrix = t.kind == Tok::KwMatrix;
                let (mut scalar, mut rows, mut cols) = (5 * 16, if matrix { 4 } else { 1 }, 4);
                if self.lx.tok.kind == Tok::Lt {
                    self.bump();
                    if self.lx.tok.kind != Tok::TyNum || self.lx.tok.aux != lex::num_scalar(self.lx.tok.aux) {
                        return Err(self.unexpected("a scalar type"));
                    }
                    scalar = self.lx.tok.aux;
                    self.bump();
                    self.expect(Tok::Comma, "','")?;
                    if matrix {
                        rows = self.template_int()? as u64;
                        self.expect(Tok::Comma, "','")?;
                    }
                    cols = self.template_int()? as u64;
                    self.close_template()?;
                }
                // one row in a matrix, one column in a vector: types of their own to DXC,
                // which the tree cannot tell from a vector and a scalar
                let shaped = if (matrix && rows == 1) || (!matrix && cols == 1) { None } else { lex::num_shaped(scalar, rows, cols) };
                match shaped {
                    Some(aux) => Ok(self.num_type(aux)),
                    None => Err(self.unsupported(format!("{} of {rows} x {cols}", if matrix { "matrix" } else { "vector" }))),
                }
            }
            _ => Err(self.unexpected("a type")),
        }
    }

    fn template_int(&mut self) -> R<u32> {
        if self.lx.tok.kind != Tok::Int || self.lx.tok.val > u32::MAX as u64 {
            return Err(self.unexpected("an integer"));
        }
        let v = self.lx.tok.val as u32;
        self.bump();
        Ok(v)
    }

    /// The `>` of a template argument list, which the lexer may have joined to another `>`.
    fn close_template(&mut self) -> R<()> {
        match self.lx.tok.kind {
            Tok::Gt => self.bump(),
            Tok::Shr => self.lx.tok.kind = Tok::Gt,
            _ => return Err(self.unexpected("'>'")),
        }
        Ok(())
    }

    /// `[n][m]` after a declared name.
    fn dims(&mut self) -> R<Vec<ExprId>> {
        let mut dims = Vec::new();
        while self.lx.tok.kind == Tok::LBracket {
            self.bump();
            if self.lx.tok.kind == Tok::RBracket {
                return Err(self.unsupported("array without a size".into()));
            }
            dims.push(self.expr()?);
            self.expect(Tok::RBracket, "']'")?;
        }
        Ok(dims)
    }

    /// `: SEMANTIC` and `: register(u0, space1)` after a declared name.
    fn bindings(&mut self) -> R<(Option<Sym>, Option<Register>)> {
        let (mut semantic, mut register) = (None, None);
        while self.lx.tok.kind == Tok::Colon {
            self.bump();
            match self.lx.tok.kind {
                Tok::Ident => {
                    semantic = Some(Sym(self.lx.tok.val as u32));
                    self.bump();
                }
                Tok::KwRegister => {
                    self.bump();
                    self.expect(Tok::LParen, "'('")?;
                    let (class, index) = self.register_part(false)?;
                    let mut space = 0;
                    match self.lx.tok.kind {
                        Tok::Comma => {
                            self.bump();
                            space = self.register_part(true)?.1;
                        }
                        Tok::LBracket => return Err(self.unsupported("register with a subcomponent".into())),
                        _ => {}
                    }
                    self.expect(Tok::RParen, "')'")?;
                    register = Some(Register { class, index, space });
                }
                _ => return Err(self.unexpected("a semantic or register")),
            }
        }
        Ok((semantic, register))
    }

    /// `u3` (class and index) or `space1`.
    fn register_part(&mut self, space: bool) -> R<(u8, u32)> {
        if self.lx.tok.kind != Tok::Ident {
            return Err(self.unexpected("a register"));
        }
        let name = self.u.syms.name(Sym(self.lx.tok.val as u32));
        let (class, digits) = if space {
            (0, name.strip_prefix("space"))
        } else {
            let c = name.as_bytes()[0].to_ascii_lowercase();
            (c, name.get(1..))
        };
        let index = digits.filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())).and_then(|d| d.parse::<u32>().ok());
        let Some(index) = index else {
            let what = format!("register({name})");
            return Err(if !space && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') && !matches!(class, b't' | b'u' | b's' | b'b') {
                self.unsupported(what)
            } else {
                self.source(format!("malformed {what}"))
            });
        };
        if !space && !matches!(class, b't' | b'u' | b's' | b'b') {
            let what = format!("register({name})");
            return Err(self.unsupported(what));
        }
        self.bump();
        Ok((class, index))
    }

    /// The declarators of a declaration and its `;`, into [`Unit::vars`].
    fn declarators(&mut self, quals: u32, ty: TypeId) -> R<()> {
        let line = self.lx.tok.line;
        let name = self.decl_name()?;
        self.declarators_from(quals, ty, name, line)
    }

    fn declarators_from(&mut self, quals: u32, ty: TypeId, mut name: Sym, mut line: u32) -> R<()> {
        loop {
            let dims = if self.lx.tok.kind == Tok::LBracket { self.dims()? } else { Vec::new() };
            let (semantic, register) = if self.lx.tok.kind == Tok::Colon { self.bindings()? } else { (None, None) };
            let mut init = None;
            if self.lx.tok.kind == Tok::Assign {
                self.bump();
                init = Some(if self.lx.tok.kind == Tok::LBrace { self.init_list()? } else { self.assign()? });
            }
            self.u.vars.push(VarDecl { name, ty, dims, init, quals, semantic, register, line });
            match self.lx.tok.kind {
                Tok::Semi => {
                    self.bump();
                    return Ok(());
                }
                Tok::Comma => {
                    self.bump();
                    line = self.lx.tok.line;
                    name = self.decl_name()?;
                }
                Tok::LParen => return Err(self.unsupported("function declared inside a function".into())),
                _ => return Err(self.unexpected("';'")),
            }
        }
    }

    fn init_list(&mut self) -> R<ExprId> {
        self.enter()?;
        self.bump();
        let base = self.expr_stack.len();
        let mut height = 0;
        while self.lx.tok.kind != Tok::RBrace {
            let e = if self.lx.tok.kind == Tok::LBrace { self.init_list()? } else { self.assign()? };
            height = height.max(self.height);
            self.expr_stack.push(e);
            match self.lx.tok.kind {
                Tok::Comma => self.bump(),
                Tok::RBrace => {}
                _ => return Err(self.unexpected("',' or '}'")),
            }
        }
        self.bump();
        self.depth -= 1;
        self.grow(height + 1)?;
        let r = self.flush_exprs(base);
        Ok(self.push(Expr::InitList(r)))
    }

    /// `[name] [name(args)]` ...
    fn attrs(&mut self) -> R<Vec<Attr>> {
        let mut attrs = Vec::new();
        while self.lx.tok.kind == Tok::LBracket {
            self.bump();
            if self.lx.tok.kind == Tok::LBracket {
                return Err(self.unsupported("attribute in [[ ]]".into()));
            }
            if self.lx.tok.kind != Tok::Ident {
                return Err(self.unexpected("an attribute name"));
            }
            let name = Sym(self.lx.tok.val as u32);
            self.bump();
            let mut args = Vec::new();
            if self.lx.tok.kind == Tok::LParen {
                self.bump();
                while self.lx.tok.kind != Tok::RParen {
                    let t = self.lx.tok;
                    args.push(match t.kind {
                        Tok::Int => AttrArg::Int(t.val),
                        Tok::Float => AttrArg::Float(f64::from_bits(t.val)),
                        Tok::Str => AttrArg::Str(self.lx.str_text(t).to_string()),
                        Tok::Ident => AttrArg::Ident(Sym(t.val as u32)),
                        _ => return Err(self.unexpected("an attribute argument")),
                    });
                    self.bump();
                    match self.lx.tok.kind {
                        Tok::Comma => self.bump(),
                        Tok::RParen => {}
                        _ => return Err(self.unsupported("attribute argument that is not a literal or a name".into())),
                    }
                }
                self.bump();
            }
            self.expect(Tok::RBracket, "']'")?;
            attrs.push(Attr { name, args });
        }
        Ok(attrs)
    }

    // ---- statements ----

    #[inline]
    fn flush_stmts(&mut self, base: usize) -> Range {
        let start = self.u.stmt_lists.len() as u32;
        self.u.stmt_lists.extend_from_slice(&self.stmt_stack[base..]);
        self.stmt_stack.truncate(base);
        Range { start, len: self.u.stmt_lists.len() as u32 - start }
    }

    #[inline]
    fn flush_exprs(&mut self, base: usize) -> Range {
        let start = self.u.expr_lists.len() as u32;
        self.u.expr_lists.extend_from_slice(&self.expr_stack[base..]);
        self.expr_stack.truncate(base);
        Range { start, len: self.u.expr_lists.len() as u32 - start }
    }

    /// The statements of a block whose `{` has been read, and its `}`. `scope`: the length
    /// of `shadows` where the block's scope began.
    fn block_body(&mut self, scope: usize) -> R<Range> {
        let base = self.stmt_stack.len();
        while self.lx.tok.kind != Tok::RBrace {
            if self.lx.tok.kind == Tok::Eof {
                return Err(self.unexpected("'}'"));
            }
            let s = self.stmt()?;
            self.stmt_stack.push(s);
        }
        // before the token after the block is read: it is outside the scope
        self.shadows.truncate(scope);
        self.bump();
        Ok(self.flush_stmts(base))
    }

    fn decl_stmt(&mut self, line: u32) -> R<StmtId> {
        let quals = self.quals();
        let ty = self.type_spec()?;
        if self.lx.tok.kind == Tok::LParen {
            return Err(self.unsupported("statement that starts with a constructor".into()));
        }
        let start = self.u.vars.len() as u32;
        self.declarators(quals, ty)?;
        Ok(self.push_stmt(Stmt::Var(Range { start, len: self.u.vars.len() as u32 - start }), line))
    }

    fn paren_expr(&mut self) -> R<ExprId> {
        self.expect(Tok::LParen, "'('")?;
        let e = self.expr()?;
        self.expect(Tok::RParen, "')'")?;
        Ok(e)
    }

    fn stmt(&mut self) -> R<StmtId> {
        let line = self.lx.tok.line;
        match self.lx.tok.kind {
            Tok::Ident | Tok::TyNum | Tok::TyRes | Tok::TySampler | Tok::TyStream | Tok::TyPatch | Tok::KwVector | Tok::KwMatrix => {
                if self.at_local_decl() {
                    return self.decl_stmt(line);
                }
            }
            Tok::LBrace => {
                self.enter()?;
                self.bump();
                let r = self.block_body(self.shadows.len())?;
                self.depth -= 1;
                return Ok(self.push_stmt(Stmt::Block(r), line));
            }
            Tok::KwIf => {
                self.enter()?;
                self.bump();
                let cond = self.paren_expr()?;
                let then = self.stmt()?;
                let mut els = None;
                if self.lx.tok.kind == Tok::KwElse {
                    self.bump();
                    els = Some(self.stmt()?);
                }
                self.depth -= 1;
                return Ok(self.push_stmt(Stmt::If { cond, then, els }, line));
            }
            Tok::KwFor => {
                self.enter()?;
                self.bump();
                self.expect(Tok::LParen, "'('")?;
                let init = if self.lx.tok.kind == Tok::Semi {
                    self.bump();
                    None
                } else if self.at_local_decl() {
                    Some(self.decl_stmt(line)?)
                } else {
                    let e = self.expr()?;
                    self.expect(Tok::Semi, "';'")?;
                    Some(self.push_stmt(Stmt::Expr(e), line))
                };
                let cond = if self.lx.tok.kind == Tok::Semi { None } else { Some(self.expr()?) };
                self.expect(Tok::Semi, "';'")?;
                let step = if self.lx.tok.kind == Tok::RParen { None } else { Some(self.expr()?) };
                self.expect(Tok::RParen, "')'")?;
                let body = self.stmt()?;
                self.depth -= 1;
                return Ok(self.push_stmt(Stmt::For { init, cond, step, body }, line));
            }
            Tok::KwWhile => {
                self.enter()?;
                self.bump();
                let cond = self.paren_expr()?;
                let body = self.stmt()?;
                self.depth -= 1;
                return Ok(self.push_stmt(Stmt::While { cond, body }, line));
            }
            Tok::KwDo => {
                self.enter()?;
                self.bump();
                let body = self.stmt()?;
                self.expect(Tok::KwWhile, "'while'")?;
                let cond = self.paren_expr()?;
                self.expect(Tok::Semi, "';'")?;
                self.depth -= 1;
                return Ok(self.push_stmt(Stmt::DoWhile { body, cond }, line));
            }
            Tok::KwSwitch => return self.switch(line),
            Tok::KwBreak | Tok::KwContinue | Tok::KwDiscard => {
                let s = match self.lx.tok.kind {
                    Tok::KwBreak => Stmt::Break,
                    Tok::KwContinue => Stmt::Continue,
                    _ => Stmt::Discard,
                };
                self.bump();
                self.expect(Tok::Semi, "';'")?;
                return Ok(self.push_stmt(s, line));
            }
            Tok::KwReturn => {
                self.bump();
                let value = if self.lx.tok.kind == Tok::Semi { None } else { Some(self.expr()?) };
                self.expect(Tok::Semi, "';'")?;
                return Ok(self.push_stmt(Stmt::Return(value), line));
            }
            Tok::Semi => {
                self.bump();
                return Ok(self.push_stmt(Stmt::Empty, line));
            }
            Tok::LBracket => {
                let attrs = self.attrs()?;
                self.enter()?;
                let s = self.stmt()?;
                self.depth -= 1;
                self.u.stmt_attrs.insert(s.0, attrs);
                return Ok(s);
            }
            Tok::KwStruct => return Err(self.unsupported("struct declared inside a function".into())),
            Tok::KwCase | Tok::KwDefault => return Err(self.source("case label outside the body of a switch".into())),
            Tok::KwElse => return Err(self.source("else without if".into())),
            _ => {}
        }
        let e = self.expr()?;
        if self.lx.tok.kind != Tok::Semi {
            // `Name x`: a declaration whose type is not one of the dialect's
            if let (Expr::Ident(s), Tok::Ident | Tok::Lt) = (self.u.exprs[e.0 as usize], self.lx.tok.kind) {
                let what = format!("type {}", self.u.syms.name(s));
                return Err(self.unsupported(what));
            }
            return Err(self.unexpected("';'"));
        }
        self.bump();
        Ok(self.push_stmt(Stmt::Expr(e), line))
    }

    fn switch(&mut self, line: u32) -> R<StmtId> {
        self.enter()?;
        self.bump();
        let value = self.paren_expr()?;
        self.expect(Tok::LBrace, "'{'")?;
        let case_base = self.case_stack.len();
        let stmt_base = self.stmt_stack.len();
        let scope = self.shadows.len();
        // the label of the case being collected (None: no label yet)
        let mut open: Option<Option<ExprId>> = None;
        loop {
            let label = match self.lx.tok.kind {
                Tok::RBrace => None,
                Tok::KwCase => {
                    self.bump();
                    let v = self.assign()?;
                    self.expect(Tok::Colon, "':'")?;
                    Some(Some(v))
                }
                Tok::KwDefault => {
                    self.bump();
                    self.expect(Tok::Colon, "':'")?;
                    Some(None)
                }
                Tok::Eof => return Err(self.unexpected("'}'")),
                _ => {
                    if open.is_none() {
                        return Err(self.source("statement before the first case of a switch".into()));
                    }
                    let s = self.stmt()?;
                    self.stmt_stack.push(s);
                    continue;
                }
            };
            if let Some(value) = open {
                let body = self.flush_stmts(stmt_base);
                self.case_stack.push(Case { value, body });
            }
            match label {
                Some(l) => open = Some(l),
                None => break,
            }
        }
        self.shadows.truncate(scope);
        self.bump();
        self.depth -= 1;
        let start = self.u.cases.len() as u32;
        self.u.cases.extend_from_slice(&self.case_stack[case_base..]);
        self.case_stack.truncate(case_base);
        let cases = Range { start, len: self.u.cases.len() as u32 - start };
        Ok(self.push_stmt(Stmt::Switch { value, cases }, line))
    }

    // ---- expressions ----

    /// An expression with the comma operator.
    fn expr(&mut self) -> R<ExprId> {
        let e = self.assign()?;
        if self.lx.tok.kind == Tok::Comma {
            return self.comma_tail(e);
        }
        Ok(e)
    }

    fn comma_tail(&mut self, mut l: ExprId) -> R<ExprId> {
        let mut height = self.height;
        while self.lx.tok.kind == Tok::Comma {
            self.bump();
            let r = self.assign()?;
            height = height.max(self.height) + 1;
            self.grow(height)?;
            l = self.push(Expr::Binary(BinOp::Comma, l, r));
        }
        Ok(l)
    }

    /// An assignment expression: what an argument or an initialiser is.
    #[inline]
    fn assign(&mut self) -> R<ExprId> {
        self.enter()?;
        let l = self.unary()?;
        let e = self.assign_tail(l)?;
        self.depth -= 1;
        Ok(e)
    }

    /// The rest of an assignment expression whose first unary operand is `l`.
    #[inline(always)]
    fn assign_tail(&mut self, l: ExprId) -> R<ExprId> {
        // most operands are an argument or an initialiser as they are
        if !CONTINUES[self.lx.tok.kind as usize] {
            return Ok(l);
        }
        self.assign_ops(l)
    }

    #[inline(never)]
    fn assign_ops(&mut self, l: ExprId) -> R<ExprId> {
        let c = if BINARY[self.lx.tok.kind as usize].0 != 0 { self.binary_tail(l, 1)? } else { l };
        let height = self.height;
        let op = match self.lx.tok.kind {
            Tok::Question => {
                self.bump();
                let t = self.expr()?;
                let height = height.max(self.height);
                self.expect(Tok::Colon, "':'")?;
                let f = self.assign()?;
                self.grow(height.max(self.height) + 1)?;
                return Ok(self.push(Expr::Ternary(c, t, f)));
            }
            Tok::Assign => None,
            Tok::PlusEq => Some(BinOp::Add),
            Tok::MinusEq => Some(BinOp::Sub),
            Tok::StarEq => Some(BinOp::Mul),
            Tok::SlashEq => Some(BinOp::Div),
            Tok::PercentEq => Some(BinOp::Rem),
            Tok::AmpEq => Some(BinOp::BitAnd),
            Tok::PipeEq => Some(BinOp::BitOr),
            Tok::CaretEq => Some(BinOp::BitXor),
            Tok::ShlEq => Some(BinOp::Shl),
            Tok::ShrEq => Some(BinOp::Shr),
            _ => return Ok(c),
        };
        self.bump();
        let r = self.assign()?;
        self.grow(height.max(self.height) + 1)?;
        Ok(self.push(Expr::Assign(op, c, r)))
    }

    /// Precedence climbing: the binary operators of precedence `min` and above after `l`.
    fn binary_tail(&mut self, mut l: ExprId, min: u8) -> R<ExprId> {
        let mut height = self.height;
        loop {
            let (prec, op) = BINARY[self.lx.tok.kind as usize];
            if prec < min || prec == 0 {
                self.height = height;
                return Ok(l);
            }
            self.bump();
            let mut r = self.unary()?;
            if BINARY[self.lx.tok.kind as usize].0 > prec {
                self.enter()?;
                r = self.binary_tail(r, prec + 1)?;
                self.depth -= 1;
            }
            height = height.max(self.height) + 1;
            self.grow(height)?;
            l = self.push(Expr::Binary(op, l, r));
        }
    }

    /// A unary expression. A name, a call and an integer are read here, in a function that is
    /// small enough to need next to no stack frame; everything else by [`Parser::unary_any`].
    fn unary(&mut self) -> R<ExprId> {
        let t = self.lx.tok;
        let e = match t.kind {
            Tok::Ident => {
                self.bump();
                let name = Sym(t.val as u32);
                if self.lx.tok.kind == Tok::LParen {
                    return self.call(name);
                }
                self.height = 1;
                self.push(Expr::Ident(name))
            }
            Tok::Int => {
                self.bump();
                self.height = 1;
                self.push(Expr::Int(t.val, lex::int_suffix(t.aux)))
            }
            _ => return self.unary_any(),
        };
        self.postfix(e)
    }

    /// The call of `name` at the current `(`.
    #[inline(never)]
    fn call(&mut self, name: Sym) -> R<ExprId> {
        let args = self.args()?;
        self.grow(self.height + 1)?;
        let e = self.push(Expr::Call(name, args));
        self.postfix(e)
    }

    #[inline(never)]
    fn unary_any(&mut self) -> R<ExprId> {
        let op = match self.lx.tok.kind {
            Tok::Minus => UnOp::Neg,
            Tok::Plus => UnOp::Plus,
            Tok::Bang => UnOp::Not,
            Tok::Tilde => UnOp::BitNot,
            Tok::PlusPlus => UnOp::PreInc,
            Tok::MinusMinus => UnOp::PreDec,
            _ => {
                let p = self.primary()?;
                return self.postfix(p);
            }
        };
        self.bump();
        let e = self.operand()?;
        self.grow(self.height + 1)?;
        Ok(self.push(Expr::Unary(op, e)))
    }

    /// The operand of a prefix operator or a cast.
    fn operand(&mut self) -> R<ExprId> {
        self.enter()?;
        let e = self.unary()?;
        self.depth -= 1;
        Ok(e)
    }

    /// `(args)` at the current `(`; leaves the height of the tallest argument.
    fn args(&mut self) -> R<Range> {
        let base = self.expr_stack.len();
        self.args_from(base)
    }

    /// [`Parser::args`], in one list with what is on the stack above `base`.
    fn args_from(&mut self, base: usize) -> R<Range> {
        self.bump();
        let mut height = 0;
        if self.lx.tok.kind != Tok::RParen {
            loop {
                let e = self.assign()?;
                height = height.max(self.height);
                self.expr_stack.push(e);
                match self.lx.tok.kind {
                    Tok::Comma => self.bump(),
                    Tok::RParen => break,
                    _ => return Err(self.unexpected("',' or ')'")),
                }
            }
        }
        self.bump();
        self.height = height;
        Ok(self.flush_exprs(base))
    }

    /// A primary expression other than a name, a call or an integer ([`Parser::unary`] reads
    /// those).
    fn primary(&mut self) -> R<ExprId> {
        let t = self.lx.tok;
        match t.kind {
            Tok::LParen => {
                self.bump();
                let inner = self.lx.tok;
                match inner.kind {
                    Tok::TyNum | Tok::KwVector | Tok::KwMatrix => {
                        // `vector<T, n>` and `matrix<T, r, c>` are the numeric types `Tn` and `Trxc`
                        let ty = if inner.kind == Tok::TyNum {
                            self.bump();
                            self.num_type(inner.aux)
                        } else {
                            self.type_spec()?
                        };
                        match self.lx.tok.kind {
                            Tok::RParen => {
                                self.bump();
                                let e = self.operand()?;
                                self.grow(self.height + 1)?;
                                Ok(self.push(Expr::Cast(ty, e)))
                            }
                            Tok::LParen => {
                                // `(uint4(...) ...)`: an expression that starts with a constructor
                                let args = self.args()?;
                                self.grow(self.height + 1)?;
                                let c = self.push(Expr::Construct(ty, args));
                                let p = self.postfix(c)?;
                                self.enter()?;
                                let mut e = self.assign_tail(p)?;
                                self.depth -= 1;
                                if self.lx.tok.kind == Tok::Comma {
                                    e = self.comma_tail(e)?;
                                }
                                self.expect(Tok::RParen, "')'")?;
                                Ok(e)
                            }
                            _ => Err(self.unexpected("')' or '('")),
                        }
                    }
                    Tok::Ident if self.is_struct(inner.val) => {
                        self.bump();
                        self.expect(Tok::RParen, "')'")?;
                        let ty = self.intern_type(TypeRef::Named(Sym(inner.val as u32)));
                        let e = self.operand()?;
                        self.grow(self.height + 1)?;
                        Ok(self.push(Expr::Cast(ty, e)))
                    }
                    Tok::TyRes | Tok::TySampler | Tok::TyStream | Tok::TyPatch | Tok::KwVoid => {
                        let what = format!("cast to {}", self.u.syms.name(Sym(inner.val as u32)));
                        Err(self.unsupported(what))
                    }
                    _ => {
                        let e = self.expr()?;
                        self.expect(Tok::RParen, "')'")?;
                        Ok(e)
                    }
                }
            }
            Tok::TyNum | Tok::KwVector | Tok::KwMatrix => {
                let ty = if t.kind == Tok::TyNum {
                    self.bump();
                    self.num_type(t.aux)
                } else {
                    self.type_spec()?
                };
                if self.lx.tok.kind != Tok::LParen {
                    return Err(self.unexpected("'(' after a type in an expression"));
                }
                let args = self.args()?;
                self.grow(self.height + 1)?;
                Ok(self.push(Expr::Construct(ty, args)))
            }
            Tok::Float => {
                self.bump();
                self.height = 1;
                Ok(self.push(Expr::Float(f64::from_bits(t.val), lex::float_suffix(t.aux))))
            }
            Tok::KwTrue | Tok::KwFalse => {
                self.bump();
                self.height = 1;
                Ok(self.push(Expr::Bool(t.kind == Tok::KwTrue)))
            }
            Tok::Str => Err(self.unsupported("string literal in an expression".into())),
            Tok::LBrace => Err(self.source("initialiser list outside an initialiser".into())),
            Tok::TyRes | Tok::TySampler | Tok::TyStream | Tok::TyPatch => {
                let what = format!("{} in an expression", self.u.syms.name(Sym(t.val as u32)));
                Err(self.unsupported(what))
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    /// The postfix operators after the primary expression `e`.
    #[inline(always)]
    fn postfix(&mut self, e: ExprId) -> R<ExprId> {
        match self.lx.tok.kind {
            Tok::Dot | Tok::LBracket | Tok::PlusPlus | Tok::MinusMinus => self.postfix_ops(e),
            _ => Ok(e),
        }
    }

    #[inline(never)]
    fn postfix_ops(&mut self, mut e: ExprId) -> R<ExprId> {
        loop {
            let height = self.height;
            match self.lx.tok.kind {
                Tok::Dot => {
                    self.bump();
                    if !lex::is_name(self.lx.tok.kind) {
                        return Err(self.unexpected("a member name"));
                    }
                    let name = Sym(self.lx.tok.val as u32);
                    self.bump();
                    if self.lx.tok.kind == Tok::LParen {
                        let base = self.expr_stack.len();
                        self.expr_stack.push(e);
                        let all = self.args_from(base)?;
                        self.height = self.height.max(height);
                        e = self.push(Expr::Method(name, all));
                    } else {
                        e = self.push(Expr::Member(e, name));
                    }
                }
                Tok::LBracket => {
                    self.bump();
                    let i = self.expr()?;
                    self.expect(Tok::RBracket, "']'")?;
                    self.height = self.height.max(height);
                    e = self.push(Expr::Index(e, i));
                }
                Tok::PlusPlus => {
                    self.bump();
                    e = self.push(Expr::Unary(UnOp::PostInc, e));
                }
                Tok::MinusMinus => {
                    self.bump();
                    e = self.push(Expr::Unary(UnOp::PostDec, e));
                }
                _ => return Ok(e),
            }
            self.grow(self.height + 1)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::astprint;
    use super::*;

    /// The tree of a unit, one line per top-level declaration or statement.
    fn tree(src: &str) -> String {
        let u = parse(src).unwrap_or_else(|e| panic!("{src}: {e}"));
        astprint::tree(&u).lines().map(str::trim).collect::<Vec<_>>().join("\n")
    }

    /// The trees of statements in a function.
    fn stmts(body: &str) -> String {
        let t = tree(&format!("void f() {{ {body} }}"));
        t.strip_prefix("(func f void (params)\n").and_then(|t| t.strip_suffix("\n)")).unwrap_or_else(|| panic!("{t}")).to_string()
    }

    fn unsupported(src: &str) -> String {
        match parse(src) {
            Err(Error::Unsupported(what)) => what,
            other => panic!("{src}: {:?}", other.map(|_| ())),
        }
    }

    fn source_error(src: &str) -> (u32, String) {
        match parse(src) {
            Err(Error::Source { line, msg }) => (line, msg),
            other => panic!("{src}: {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(stmts("a = b + c * d - e;"), "(Assign a (Sub (Add b (Mul c d)) e))");
        assert_eq!(stmts("a = b * c / d % e;"), "(Assign a (Rem (Div (Mul b c) d) e))");
        assert_eq!(
            stmts("a = b << 1 + 2 & 3 == 4 | 5 ^ 6 && 7 || 8;"),
            "(Assign a (LogOr (LogAnd (BitOr (BitAnd (Shl b (Add 1:None 2:None)) (Eq 3:None 4:None)) (BitXor 5:None 6:None)) 7:None) 8:None))"
        );
        assert_eq!(stmts("a = b < c == d > e != f <= g >= h;"), "(Assign a (Ne (Eq (Lt b c) (Gt d e)) (Ge (Le f g) h)))");
        assert_eq!(stmts("a = b || c && d | e ^ f & g == h < i << j + k * l;"), "(Assign a (LogOr b (LogAnd c (BitOr d (BitXor e (BitAnd f (Eq g (Lt h (Shl i (Add j (Mul k l)))))))))))");
        assert_eq!(stmts("a = b - c - d; a = b >> c >> d; a = b / c * d;"), "(Assign a (Sub (Sub b c) d))\n(Assign a (Shr (Shr b c) d))\n(Assign a (Mul (Div b c) d))");
        assert_eq!(stmts("a = b ? c : d ? e : f; a = b = c; a += b -= c; a = c ? x : y = z;"), "(Assign a (? b c (? d e f)))\n(Assign a (Assign b c))\n(AssignAdd a (AssignSub b c))\n(Assign a (? c x (Assign y z)))");
        assert_eq!(stmts("a = b ? c, d : e; a = (b, c), d;"), "(Assign a (? b (Comma c d) e))\n(Comma (Assign a (Comma b c)) d)");
        assert_eq!(stmts("a *= b; a /= b; a %= b; a &= b; a |= b; a ^= b; a <<= b; a >>= b;"), "(AssignMul a b)\n(AssignDiv a b)\n(AssignRem a b)\n(AssignBitAnd a b)\n(AssignBitOr a b)\n(AssignBitXor a b)\n(AssignShl a b)\n(AssignShr a b)");
        assert_eq!(stmts("a = -b * ~c + !d - +e; a = - -b; a = -b++; a = b++ + ++c; a-- - --b;"), "(Assign a (Sub (Add (Mul (Neg b) (BitNot c)) (Not d)) (Plus e)))\n(Assign a (Neg (Neg b)))\n(Assign a (Neg (PostInc b)))\n(Assign a (Add (PostInc b) (PreInc c)))\n(Sub (PostDec a) (PreDec b))");
    }

    #[test]
    fn casts_constructors_calls_and_members() {
        assert_eq!(stmts("a = (uint)x.y + (uint)(x) + uint(x) + (x);"), "(Assign a (Add (Add (Add (cast uint (. x y)) (cast uint x)) (new uint x)) x))");
        assert_eq!(stmts("a = -(int)x << 3; a = (float3)v.xyz; a = (uint)-x; a = (double)(int)x; a = (uint)(x) + y;"), "(Assign a (Shl (Neg (cast int x)) 3:None))\n(Assign a (cast float3 (. v xyz)))\n(Assign a (cast uint (Neg x)))\n(Assign a (cast double (cast int x)))\n(Assign a (Add (cast uint x) y))");
        assert_eq!(stmts("a = (uint)f(x)[1] * 2u; a = (uint)(((int)(v << 8)) >> 8);"), "(Assign a (Mul (cast uint ([] (call f x) 1:None)) 2:U))\n(Assign a (cast uint (Shr (cast int (Shl v 8:None)) 8:None)))");
        assert_eq!(stmts("a = b.c.d; a = b.c(d).e[f].g; a = (s == 0u).xxxx; a = b[c][d]; b.c.d(e);"), "(Assign a (. (. b c) d))\n(Assign a (. ([] (. (method b c d) e) f) g))\n(Assign a (. (Eq s 0:U) xxxx))\n(Assign a ([] ([] b c) d))\n(method (. b c) d e)");
        assert_eq!(stmts("a = (uint4(1, 2, 3, 4).x + y, z); a = (float2(x, y)); a = (uint(x) ? b : c); a = (uint2(1, 2)[i] = 3);"), "(Assign a (Comma (Add (. (new uint4 1:None 2:None 3:None 4:None) x) y) z))\n(Assign a (new float2 x y))\n(Assign a (? (new uint x) b c))\n(Assign a (Assign ([] (new uint2 1:None 2:None) i) 3:None))");
        assert_eq!(stmts("f(); f(a, g(b, (c, d)), h()); select(c, a, b); and(a, or(b, c));"), "(call f)\n(call f a (call g b (Comma c d)) (call h))\n(call select c a b)\n(call and a (call or b c))");
        assert_eq!(stmts("t = ResourceDescriptorHeap[NonUniformResourceIndex(i + 1u)]; mem.Store4(a, uint4(0u, 1u, 2u, 3u)); x = true ? 1.5f : false;"), "(Assign t ([] ResourceDescriptorHeap (call NonUniformResourceIndex (Add i 1:U))))\n(method mem Store4 a (new uint4 0:U 1:U 2:U 3:U))\n(Assign x (? true 1.5#3ff8000000000000:F false))");
        assert_eq!(tree("struct S { uint a; }; S g; void f() { S x = (S)0; x = (S)(y); a = (S) - 1; }"), "(struct S (var a uint))\n(var g S)\n(func f void (params)\n(decl (var x S = (cast S 0:None)))\n(Assign x (cast S y))\n(Assign a (cast S (Neg 1:None)))\n)");
        // a name that is no type: parentheses, not a cast
        assert_eq!(stmts("a = (S) - 1; a = (b);"), "(Assign a (Sub S 1:None))\n(Assign a b)");
        // no function values in the dialect
        assert_eq!(source_error("void f() { a = (b)(c); }").1, "expected ';', found '('");
    }

    #[test]
    fn declarations() {
        assert_eq!(
            tree("static const uint T[3] = { 1u, 2u, 3u }; uint4 v[2][3]; float2x3 m; RWByteAddressBuffer io : register(u0); SamplerState ss : register(s2, space1); groupshared uint lds[8192]; static uint a = 1u << 16, b;"),
            "(var T static const uint [3:None] = (init 1:U 2:U 3:U))\n(var v uint4 [2:None] [3:None])\n(var m float2x3)\n(var io RWByteAddressBuffer @u0,0)\n(var ss SamplerState @s2,1)\n(var lds groupshared uint [8192:None])\n(var a static uint = (Shl 1:U 16:None))\n(var b static uint)"
        );
        assert_eq!(
            tree("struct S { float4 pos : SV_Position; nointerpolation uint4 a[2], b; noperspective centroid float c : X0; } s0, s1[2];"),
            "(struct S (var pos float4 :SV_Position) (var a nointerpolation uint4 [2:None]) (var b nointerpolation uint4) (var c noperspective centroid float :X0))\n(var s0 S)\n(var s1 S [2:None])"
        );
        assert_eq!(
            stmts("uint a = 1u, b[2], c = a; precise float p = a * b; float tn[4]; uint2 pairs[2] = { uint2(0, 1), { 2, 3 }, }; const bool k = true; globallycoherent RWByteAddressBuffer w = ResourceDescriptorHeap[0];"),
            "(decl (var a uint = 1:U) (var b uint [2:None]) (var c uint = a))\n(decl (var p precise float = (Mul a b)))\n(decl (var tn float [4:None]))\n(decl (var pairs uint2 [2:None] = (init (new uint2 0:None 1:None) (init 2:None 3:None))))\n(decl (var k const bool = true))\n(decl (var w globallycoherent RWByteAddressBuffer = ([] ResourceDescriptorHeap 0:None)))"
        );
        assert_eq!(
            stmts("Texture2D<float4> t; Texture2DMS<float4, 4> m; RWTexture3D<uint> u; TextureCubeArray c; RWStructuredBuffer<uint> s; SamplerComparisonState q; vector<float, 3> v; matrix<int, 2, 3> x; vector w; matrix y; Buffer<vector<uint, 4>> z;"),
            "(decl (var t Texture2D<float4>))\n(decl (var m Texture2DMS<float4, 4>))\n(decl (var u RWTexture3D<uint>))\n(decl (var c TextureCubeArray))\n(decl (var s RWStructuredBuffer<uint>))\n(decl (var q SamplerComparisonState))\n(decl (var v float3))\n(decl (var x int2x3))\n(decl (var w float4))\n(decl (var y float4x4))\n(decl (var z Buffer<uint4>))"
        );
        assert_eq!(
            tree("struct V { float4 p : SV_Position; };\n[numthreads(8, 4, 1)] [WaveSize(32)]\n[outputtopology(\"triangle\")] void main(uint3 gid : SV_GroupID, in uint a, out vertices V verts[3], out indices uint3 tris[1], inout TriangleStream<V> s, triangle V v[3], in payload V p, InputPatch<V, 3> ip) { }\nfloat4 ps(float4 p : SV_Position, sample float2 uv : TEXCOORD0) : SV_Target0 { return p; }\nuint g(void); static inline uint h();"),
            "(struct V (var p float4 :SV_Position))\n(func main void [numthreads(8, 4, 1)] [WaveSize(32)] [outputtopology(\"triangle\")] (params (var gid uint3 :SV_GroupID) (var a in uint) (var verts out vertices V [3:None]) (var tris out indices uint3 [1:None]) (var s inout TriangleStream<V>) (var v triangle V [3:None]) (var p in payload V) (var ip InputPatch<V, 3>))\n)\n(func ps float4 :SV_Target0 (params (var p float4 :SV_Position) (var uv sample float2 :TEXCOORD0))\n(return p)\n)\n(func g uint (params) prototype)\n(func h uint (params) prototype)"
        );
        // synonyms are one type, and each type has one id
        let u = parse("uint a; dword b; uint32_t c; dword d; float4 e; vector f; vector<float, 4> g; Texture2D<float4> h; Texture2D<vector<float, 4> > i; Texture2D j;").unwrap();
        let ids: Vec<u32> = u.vars.iter().map(|v| v.ty.0).collect();
        assert_eq!(ids, [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]);
        assert_eq!(u.types.len(), 4);
        assert_eq!(u.vars.iter().map(|v| v.line).collect::<Vec<_>>(), [1; 10]);
    }

    #[test]
    fn statements() {
        assert_eq!(
            stmts("switch (x) { case 1u: case 2u: a = 1; break; default: b = 2; case 3u: { c = 3; } }"),
            "(switch x\n(case 1:U\n)\n(case 2:U\n(Assign a 1:None)\nbreak\n)\n(default\n(Assign b 2:None)\n)\n(case 3:U\n(block\n(Assign c 3:None)\n)\n)\n)"
        );
        assert_eq!(stmts("switch (x) { }  switch (y) { default: switch (z) { case 0: break; } }"), "(switch x\n)\n(switch y\n(default\n(switch z\n(case 0:None\nbreak\n)\n)\n)\n)");
        assert_eq!(
            stmts("[unroll] for (uint i = 0u, j = 1u; i < 4u; i++) { if (i == 1u) continue; else if (i == 2u) break; } [loop] [fastopt] for (;;) ; for (i = 0; ; i += 2) discard;"),
            "[unroll] (for\n(decl (var i uint = 0:U) (var j uint = 1:U))\n(Lt i 4:U) ; (PostInc i)\n(block\n(if (Eq i 1:U)\ncontinue\nelse\n(if (Eq i 2:U)\nbreak\n)\n)\n)\n)\n[loop] [fastopt] (for\n-\n- ; -\nempty\n)\n(for\n(Assign i 0:None)\n- ; (AssignAdd i 2:None)\ndiscard\n)"
        );
        assert_eq!(stmts("do { a++; } while (a < 3); while (a) a--; do a++; while (a); { } ; return; return a, b;"), "(do (Lt a 3:None)\n(block\n(PostInc a)\n)\n)\n(while a\n(PostDec a)\n)\n(do a\n(PostInc a)\n)\n(block\n)\nempty\n(return)\n(return (Comma a b))");
        // the else belongs to the nearest if
        assert_eq!(stmts("if (a) if (b) c; else d;"), "(if a\n(if b\nc\nelse\nd\n)\n)");
        assert_eq!(stmts("[branch] if (a) { } [flatten] if (b) c; [allow_uav_condition] while (d) { } [forcecase] switch (e) { } [unroll(4)] for (;;) { }"), "[branch] (if a\n(block\n)\n)\n[flatten] (if b\nc\n)\n[allow_uav_condition] (while d\n(block\n)\n)\n[forcecase] (switch e\n)\n[unroll(4)] (for\n-\n- ; -\n(block\n)\n)");
        // ids, lines and attributes of statements
        let u = parse("void f()\n{\n    a = 1;\n    [loop]\n    for (;;) {\n        b = 2;\n    }\n}\n").unwrap();
        let body = u.block(u.funcs[0].body.unwrap());
        assert_eq!(body.len(), 2);
        let Stmt::For { body: inner, .. } = *u.stmt(body[1]) else { panic!() };
        assert!(inner.0 < body[1].0);
        assert_eq!(u.stmt_attrs.keys().copied().collect::<Vec<_>>(), [body[1].0]);
        assert_eq!((u.stmt_lines[body[0].0 as usize], u.stmt_lines[body[1].0 as usize], u.stmt_lines[inner.0 as usize], u.funcs[0].line), (3, 5, 5, 1));
    }

    #[test]
    fn words_that_are_names_where_declared() {
        // the translator's prelude: a local named like a type
        assert_eq!(
            tree("void f() { uint64_t half = 1ull; a = half + (half) - half; if (fraction > half || (fraction == half)) { a = half; } } void g() { half x = (half)1; }"),
            "(func f void (params)\n(decl (var half uint64_t = 1:ULL))\n(Assign a (Sub (Add half half) half))\n(if (LogOr (Gt fraction half) (Eq fraction half))\n(block\n(Assign a half)\n)\n)\n)\n(func g void (params)\n(decl (var x half = (cast half 1:None)))\n)"
        );
        // inside the scope of the local, the word is no type
        assert_eq!(unsupported("void f() { uint half = 1u; half x; }"), "type half");
        assert_eq!(source_error("void f() { uint half = 1u; a = (half)x; }").1, "expected ';', found 'x'");
        // scopes: a block, a parameter, a for statement's enclosing block, the unit
        assert_eq!(stmts("{ uint uint4 = 1u; a = uint4; } uint4 b = uint4(1, 2, 3, 4);"), "(block\n(decl (var uint4 uint = 1:U))\n(Assign a uint4)\n)\n(decl (var b uint4 = (new uint4 1:None 2:None 3:None 4:None)))");
        assert_eq!(tree("uint f(uint dword) { return dword; } dword g(dword x) { return (dword)x; }"), "(func f uint (params (var dword uint))\n(return dword)\n)\n(func g uint (params (var x uint))\n(return (cast uint x))\n)");
        assert_eq!(tree("uint f(uint dword); dword d;"), "(func f uint (params (var dword uint)) prototype)\n(var d uint)");
        assert_eq!(tree("static uint sample; float sample2(float precise) { float p = precise; sample = 1u; return p; }"), "(var sample static uint)\n(func sample2 float (params (var precise float))\n(decl (var p float = precise))\n(Assign sample 1:U)\n(return p)\n)");
        assert_eq!(tree("uint sample(uint x) { return x; } void f() { sample(1u); }").lines().nth(4), Some("(call sample 1:U)"));
        // fields may use the words, and hide nothing
        assert_eq!(tree("struct S { uint half; float sample; }; void f() { S s; s.half = 1u; half h = s.sample; }"), "(struct S (var half uint) (var sample float))\n(func f void (params)\n(decl (var s S))\n(Assign (. s half) 1:U)\n(decl (var h half = (. s sample)))\n)");
        // what DXC reserves is no name
        for word in ["float", "int", "bool", "double", "void", "line", "point", "triangle", "linear", "centroid", "static", "const", "in", "out", "inout", "register", "struct", "if", "true", "typedef", "nointerpolation"] {
            assert!(parse(&format!("void f() {{ uint {word} = 1u; }}")).is_err(), "{word}");
            assert!(parse(&format!("void f(uint {word}) {{ }}")).is_err(), "{word}");
            assert!(parse(&format!("struct S {{ uint {word}; }};")).is_err(), "{word}");
        }
    }

    #[test]
    fn hlsl_outside_the_dialect_is_named() {
        for (src, what) in [
            ("typedef uint T;", "keyword typedef"),
            ("cbuffer C { float x; };", "keyword cbuffer"),
            ("void f() { a = b::c; }", "scope operator ::"),
            ("void f() { goto x; }", "keyword goto"),
            ("void f() { a = sizeof(x); }", "keyword sizeof"),
            ("void f() { a.b->c; }", "operator ->"),
            ("template<typename T> T f(T x) { return x; }", "keyword template"),
            ("namespace N { }", "keyword namespace"),
            ("class C { };", "keyword class"),
            ("void f() { unsigned int x; }", "keyword unsigned"),
            ("Texture2D<unorm float4> t;", "keyword unorm"),
            ("void f() { float1x3 m; }", "type float1x3"),
            ("void f() { uint1 m; }", "type uint1"),
            ("void f() { min10float m; }", "type min10float"),
            ("void f() { RayDesc r; }", "type RayDesc"),
            ("Foo x;", "type Foo"),
            ("void f(Foo x) { }", "type Foo"),
            ("vector<float, 5> v;", "vector of 1 x 5"),
            ("matrix<float, 1, 3> v;", "matrix of 1 x 3"),
            ("vector<uint, 1> v;", "vector of 1 x 1"),
            ("matrix<float, 2, 5> v;", "matrix of 2 x 5"),
            ("uint a[];", "array without a size"),
            ("void f(uint a[]) { }", "array without a size"),
            ("void f() { struct T { int x; }; }", "struct declared inside a function"),
            ("struct { int x; } s;", "struct without a name"),
            ("struct S;", "struct declared without a body"),
            ("struct A { int x; }; struct B : A { };", "struct inheritance"),
            ("struct S { void m() { } };", "member function of a struct"),
            ("struct S { struct T { int x; } t; };", "struct declared inside a struct"),
            ("struct S { int x = 1; };", "default value of a struct field"),
            ("void f(uint a = 1u) { }", "default value of a parameter"),
            ("void f(uint);", "parameter without a name"),
            ("void f() { uint g(uint x); }", "function declared inside a function"),
            ("precise float f() { return 1.0; }", "qualified function f"),
            ("void f() { a = 1.xxxx; }", "member access on a numeric literal"),
            ("void f() { a = \"s\"; }", "string literal in an expression"),
            ("void f() { a = (Texture2D)x; }", "cast to Texture2D"),
            ("void f() { a = Texture2D; }", "Texture2D in an expression"),
            ("void f() { uint4(1, 2, 3, 4).x; }", "statement that starts with a constructor"),
            ("void f() { [[unroll]] for (;;) { } }", "attribute in [[ ]]"),
            ("[numthreads(N + 1, 1, 1)] void f() { }", "attribute argument that is not a literal or a name"),
            ("float x : packoffset(c0);", "packoffset"),
            ("Texture2D t : register(t0[1]);", "register with a subcomponent"),
            ("Texture2D t : register(ps, t0);", "register(ps)"),
            ("Texture2D t : register(c0);", "register(c0)"),
            ("#pragma pack_matrix(row_major)\n", "preprocessor: #pragma"),
        ] {
            assert_eq!(unsupported(src), what, "{src}");
        }
    }

    #[test]
    fn text_that_is_not_hlsl_is_a_source_error_with_its_line() {
        for (src, line, msg) in [
            ("void f() {\n  a = ;\n}", 2, "expected an expression, found ';'"),
            ("void f() {\n\n  a = (1 + ; }", 3, "expected an expression, found ';'"),
            ("void f() {\n a = 3 4; }", 2, "expected ';', found a number"),
            ("void f() {", 1, "expected '}', found the end of the source"),
            ("void f() { if (a) {\n", 2, "expected '}', found the end of the source"),
            ("void f() { switch (a) {\n case 1:\n", 3, "expected '}', found the end of the source"),
            ("void f() { case 1: ; }", 1, "case label outside the body of a switch"),
            ("void f() { else ; }", 1, "else without if"),
            ("void f() { a = b ? c; }", 1, "expected ':', found ';'"),
            ("void f() { switch (x) { a = 1; } }", 1, "statement before the first case of a switch"),
            ("uint register;", 1, "expected a name, found 'register'"),
            ("void f() { uint x[2] = { 1, 2 ; }", 1, "expected ',' or '}', found ';'"),
            ("void f() { a = { 1, 2 }; }", 1, "initialiser list outside an initialiser"),
            ("\"str\"", 1, "expected a type, found a string"),
            ("void f() { for (uint i = 0u; i < 3u) { } }", 1, "expected ';', found ')'"),
            ("[numthreads(1, 1, 1)] uint x;", 1, "attribute before a declaration that is not a function"),
            ("void f() { a = b c; }", 1, "expected ';', found 'c'"),
            ("void f() { f(a,); }", 1, "expected an expression, found ')'"),
            ("void f(uint a,) { }", 1, "expected a parameter, found ')'"),
            ("void f() { a = uint; }", 1, "expected '(' after a type in an expression, found ';'"),
            ("void f() { a = (uint; }", 1, "expected ')' or '(', found ';'"),
            ("void f() { a.1; }", 1, "expected ';', found a number"),
            ("void f() { a. = 1; }", 1, "expected a member name, found '='"),
            ("void f() { do a++; until (b); }", 1, "expected 'while', found 'until'"),
            ("void f() { a[1; }", 1, "expected ']', found ';'"),
            ("void f() { }\n}", 2, "expected a type, found '}'"),
            ("uint x : register(u);", 1, "malformed register(u)"),
            ("uint x : register(u0, spaceX);", 1, "malformed register(spaceX)"),
            ("uint x : register(u0, 1);", 1, "expected a register, found a number"),
            ("Texture2D<float4 t;", 1, "expected '>', found 't'"),
            ("Texture2DMS<float4, x> t;", 1, "expected an integer, found 'x'"),
            ("vector<S, 3> v;", 1, "expected a scalar type, found 'S'"),
            ("struct S { uint a }", 1, "expected ';', found '}'"),
            ("void f() { a = 1 @ 2; }", 1, "unexpected character '@'"),
        ] {
            assert_eq!(source_error(src), (line, msg.to_string()), "{src}");
        }
    }
}
