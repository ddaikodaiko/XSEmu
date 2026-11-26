//! The parser and the printer on programs in the translator's style, on the translator's own
//! preludes (read from `crates/xcl` when it is there), and against DXC as the oracle: the
//! printed form of a tree must compile to the same DXIL as the text it was parsed from.
//! Set `XCL_DXIL_DXC` to the folder of a Linux DXC build (`bin/dxc`, `lib/`) for the oracle
//! tests; without it they pass with a note.

use std::path::{Path, PathBuf};
use std::process::Command;

use xcl_dxil::hlsl::ast::{Expr, Global, Stmt, TypeRef};
use xcl_dxil::hlsl::{astprint, parse};
use xcl_dxil::Error;

const CS: &str = r##"// a compute program in the translator's style
#define XCL_LDS_TOP 8188u
#define XCL_XS(i) xcl_lds[4096u + (i)]
#define XCL_SPIN 1
RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
static uint lane_id;
static uint xcl_budget = 1u << 16;
static bool xcl_w32 = false;
static const uint FMT_INFO[4] = { 0x00000000u, 0x00082001u, 0x3F800000u, 0xFFFFFFFFu };
groupshared uint xcl_lds[8192]; // D3D12 limit: 32 KB
struct Pair { uint lo; uint hi; };
uint mul_hi_u32(uint a, uint b) {
    uint al = a & 0xFFFFu, ah = a >> 16, bl = b & 0xFFFFu, bh = b >> 16;
    uint lh = al * bh, hl = ah * bl;
    uint mid = ((al * bl) >> 16) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    return ah * bh + (lh >> 16) + (hl >> 16) + (mid >> 16);
}
bool lanebit(uint lo, uint hi) { return ((select(lane_id < 32u, lo, hi) >> (lane_id & 31u)) & 1u) != 0u; }
uint2 ballot(bool b) {
    uint4 m = WaveActiveBallot(b);
    return uint2(select(and(xcl_w32, lane_id != 0u), m.y, m.x), select(xcl_w32, 0u, m.y));
}
float fma_legacy(float a, float b, float c) {
    precise float z = 0.0f + c;
    return (a == 0.0f || b == 0.0f) ? z : mad(a, b, c);
}
double round_d(double x) {
    uint l, h; asuint(x, l, h);
    uint64_t mag = ((uint64_t)(h & 0x7FFFFFFFu) << 32) | (uint64_t)l;
    uint64_t half = 1ull << 20;
    if (mag > half || (mag == half && (l & 1u) != 0u)) mag += 1ull << 3;
    return asdouble((uint)mag, (uint)(mag >> 32));
}
void put_bits(inout uint4 r, uint off, uint w, uint v) {
    uint m = w >= 32u ? 0xFFFFFFFFu : ((1u << w) - 1u); v &= m;
    uint i = off >> 5; uint s = off & 31u;
    r[i] |= v << s;
    if (s != 0u && s + w > 32u && i < 3u) r[i + 1u] |= v >> (32u - s);
}
uint perm(uint s0, uint s1, uint sel) {
    uint r = 0u;
    [unroll] for (uint i = 0u; i < 4u; i++) {
        uint k = (sel >> (8u * i)) & 0xFFu; uint b;
        if (k >= 13u) b = 0xFFu; else if (k == 12u) b = 0u;
        else b = ((k < 4u) ? (s1 >> (8u * k)) : (s0 >> (8u * (k - 4u)))) & 0xFFu;
        r |= b << (8u * i);
    }
    return r;
}
Pair split(uint64_t v) { Pair p; p.lo = (uint)v; p.hi = (uint)(v >> 32); return p; }
[WaveSize(32)]
[numthreads(8, 4, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    lane_id = WaveGetLaneIndex();
    uint s0 = io.Load(0u);
    uint s1 = io.Load(4u);
    uint v0 = gtid.x, v1 = gtid.y + gid.z;
    uint scc = 0u;
    uint exec_lo, exec_hi;
    { uint2 e0 = ballot(true); exec_lo = e0.x; exec_hi = e0.y; }
    uint pc = 0u;
    [loop] while (pc != 0xFFFFFFFFu && xcl_budget != 0u) {
        xcl_budget--;
        bool xcl_on = lanebit(exec_lo, exec_hi);
        if (pc == 0u) {
            // 0: v_add_u32 v1, s0, v0
            uint t1 = v0 + s0;
            v1 = select(xcl_on, (t1), v1);
            scc = (uint)(t1 != 0u);
            uint64_t t2 = (uint64_t)v0 * (uint64_t)s1 + 0xFFFFFFFFull;
            Pair p = split(t2);
            v0 = xcl_on ? p.lo ^ p.hi : v0;
            pc = 16u;
        }
        if (pc == 16u) {
            uint t3 = 0u; switch (s0 & 3u) { case 0u: t3 = v0; break; case 1u: t3 = v1; break; case 2u: case 3u: t3 = s1; default: break; }
            uint4 t4 = uint4(t3, v0, v1, 0u);
            put_bits(t4, 5u, 7u, mul_hi_u32(v0, v1));
            float t5 = fma_legacy(asfloat(v0), 2.0f, -1.5f) + (float)(int)v1 * 1e-8f;
            double t6 = round_d((double)t5);
            uint t6l, t6h; asuint(t6, t6l, t6h);
            v0 = perm(t4.x, t4.w, FMT_INFO[v1 & 3u]) + asuint(t5) + t6l + t6h;
            XCL_XS(lane_id & 31u) = v0;
            GroupMemoryBarrierWithGroupSync();
            v1 = xcl_lds[min(((v0) >> 2) & 8191u, XCL_LDS_TOP)];
#if XCL_SPIN
            uint t7; InterlockedAdd(xcl_lds[8189u], 1u, t7);
            v1 += t7;
#else
            v1 = 0u;
#endif
            pc = (s1 & 1u) != 0u ? 0u : 0xFFFFFFFFu;
        }
    }
    if (lane_id == 0u) { res.Store(0u, s0); res.Store(4u, scc); }
    res.Store((8u + gi * 2u) * 4u, v0);
    res.Store((9u + gi * 2u) * 4u, v1);
}
"##;

const PS: &str = r##"// a pixel program in the translator's style
#define XCL_PIXEL 1
RWByteAddressBuffer io : register(u0);
static uint lane_id;
struct XclIn {
    noperspective centroid float4 xcl_pos : SV_Position;
    nointerpolation uint4 a0 : TEXCOORD0;
     float4 a1 : TEXCOORD1;
    bool xcl_ff : SV_IsFrontFace;
};
struct XclOut {
    float4 t0 : SV_Target0;
    float xcl_depth : SV_DepthLessEqual;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    bool xcl_live = false;
    lane_id = WaveGetLaneIndex();
    uint s3 = io.Load(12u);
    Texture2D<float4> t1_T = ResourceDescriptorHeap[NonUniformResourceIndex(s3)];
    SamplerState t1_S = SamplerDescriptorHeap[NonUniformResourceIndex(s3 & 0xFFFu)];
    float4 t1 = t1_T.Sample(t1_S, float2(xi.a1.x, xi.a1.y));
    uint t1_w, t1_h, t1_n; t1_T.GetDimensions(0u, t1_w, t1_h, t1_n);
    RWTexture2D<uint4> t2_U = ResourceDescriptorHeap[NonUniformResourceIndex(s3 + 1u)];
    t2_U[uint2(xi.xcl_pos.xy)] = xi.a0;
    uint q0 = 0u;
#ifdef XCL_PIXEL
    q0 = QuadReadLaneAt(asuint(t1.x), 0u) /* helper lanes */ + lane_id;
#endif
    xo.t0 = t1 * (xi.xcl_ff ? 1.0f : 0.5f) + (float)(t1_w + t1_h + t1_n + q0);
    xo.xcl_depth = xi.xcl_pos.z;
    xcl_live = t1.w > 0.0f;
    if (!xcl_live) discard;
    return xo;
}
"##;

const GS: &str = r##"// a console rect list's rectangles
struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    uint4 p1 : TEXCOORD1;
};
[maxvertexcount(6)]
void main(triangle XclOut v[3], inout TriangleStream<XclOut> s)
{
    uint c = 3u;
    if (v[0].pos.w > 0.0 && v[1].pos.w == v[0].pos.w) {
        float2 q[3] = { v[0].pos.xy, v[1].pos.xy, v[2].pos.xy };
        [unroll] for (uint i = 0u; i < 3u; i++) {
            float2 p = q[i], a = q[(i + 1u) % 3u], b = q[(i + 2u) % 3u];
            bool corner = (p.x == a.x && p.y == b.y) || (p.x == b.x && p.y == a.y);
            bool degenerate = all(p == a) || all(p == b) || all(a == b);
            if (c == 3u && corner && !degenerate)
                c = i;
        }
    }
    if (c == 3u) {
        s.Append(v[0]);
        s.Append(v[1]);
        s.Append(v[2]);
        return;
    }
    XclOut k = v[c], a = v[(c + 1u) % 3u], b = v[(c + 2u) % 3u];
    XclOut d = v[0];
    d.pos = a.pos + b.pos - k.pos;
    d.p0 = a.p0 + b.p0 - k.p0;
    s.Append(v[0]);
    s.Append(v[1]);
    if (c == 0u)
        s.Append(d);
    else
        s.Append(v[2]);
    s.RestartStrip();
}
"##;

const MS: &str = r##"// mesh pass-through
RWByteAddressBuffer res : register(u1);
struct XclVert {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float2 p1 : SV_ClipDistance0;
};
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID, out vertices XclVert verts[64], out indices uint3 tris[32])
{
    uint rec = gid.x * 1024u;
    uint nv = min(res.Load(rec * 4u), 64u);
    uint np = min(res.Load(rec * 4u + 4u), 32u);
    SetMeshOutputCounts(nv, np);
    for (uint i = gi; i < nv; i += 32u) {
        uint b = (rec + 2u + i * 12u) * 4u;
        XclVert o;
        o.pos = asfloat(res.Load4(b));
        o.p0 = asfloat(res.Load4(b + 16u).xyzw);
        o.p1 = asfloat(res.Load4(b + 32u).xy);
        verts[i] = o;
    }
    for (uint j = gi; j < np; j += 32u) {
        uint p = res.Load((rec + 770u + j) * 4u);
        uint3 x = uint3(p & 0x3FFu, (p >> 10) & 0x3FFu, (p >> 20) & 0x3FFu);
        bool bad = or((p >> 31) != 0u, any(x >= nv));
        tris[j] = select(bad, uint3(0u, 0u, 0u), x);
    }
}
"##;

const VS: &str = r##"// a vertex program in the translator's style
RWByteAddressBuffer io : register(u0);
static uint lane_id;
struct XclIn {
    float3 a0 : POSITION0;
    uint2 a1_1 : BLENDINDICES0;
    uint xcl_vid : SV_VertexID;
    uint xcl_iid : SV_InstanceID;
};
struct XclOut {
    float4 pos : SV_Position;
    float4 p0 : TEXCOORD0;
    float p1 : SV_ClipDistance0;
};
XclOut main(XclIn xi)
{
    XclOut xo = (XclOut)0;
    lane_id = WaveGetLaneIndex();
    uint s2 = io.Load(8u);
    uint v0 = xi.xcl_vid;
    uint v3 = xi.xcl_iid;
    uint v4 = asuint(xi.a0[0]), v5 = asuint(xi.a0[1]), v6 = asuint(xi.a0[2]);
    uint v7 = xi.a1_1[1];
    do { v0 += v7; v7 >>= 1; } while (v7 != 0u);
    for (uint k = 0u; ; ++k) { if (k >= (s2 & 3u)) break; v3 = v3 * 5u + 1u; }
    xo.pos = float4(asfloat(v4), asfloat(v5), asfloat(v6), 1.0f);
    xo.p0 = float4((float)v0, (float)v3, 0.25, -0.0f);
    xo.p1 = (float)(int)s2 * 4294967295.0f;
    return xo;
}
"##;

const SAMPLES: [(&str, &str); 5] = [("cs_6_6", CS), ("ps_6_6", PS), ("gs_6_6", GS), ("ms_6_6", MS), ("vs_6_6", VS)];

/// parse(print(parse(x))) is the tree of parse(x); returns the printed text.
fn round_trip(what: &str, src: &str) -> String {
    let a = parse::parse(src).unwrap_or_else(|e| panic!("{what}: {e}"));
    let printed = astprint::unit(&a);
    let b = parse::parse(&printed).unwrap_or_else(|e| panic!("{what}: the printed text does not parse: {e}\n{printed}"));
    assert!(astprint::tree(&a) == astprint::tree(&b), "{what}: the printed text parses to another tree\n{printed}");
    assert_eq!(printed, astprint::unit(&b), "{what}: printing is not a fixed point");
    printed
}

#[test]
fn samples_round_trip() {
    for (profile, src) in SAMPLES {
        round_trip(profile, src);
    }
}

#[test]
fn sample_trees_have_what_was_written() {
    let u = parse::parse(CS).unwrap();
    // macros are expanded, the unselected `#else` branch is gone
    let tree = astprint::tree(&u);
    assert!(tree.contains("(Assign ([] xcl_lds (Add 4096:U (BitAnd lane_id 31:U))) v0)"), "{tree}");
    assert!(tree.contains("(call min (BitAnd (Shr v0 2:None) 8191:U) 8188:U)"));
    assert!(tree.contains("(call InterlockedAdd ([] xcl_lds 8189:U) 1:U t7)") && !tree.contains("(Assign v1 0:U)"));
    assert!(!tree.contains("XCL_"));
    // a local named like a type, as the translator's prelude has
    assert!(tree.contains("(decl (var half uint64_t = (Shl 1:ULL 20:None)))"));
    assert!(tree.contains("(LogOr (Gt mag half) (LogAnd (Eq mag half)"));
    let main = u.funcs.last().unwrap();
    assert_eq!(u.name(main.name), "main");
    assert_eq!(main.attrs.len(), 2);
    assert_eq!(main.params.iter().map(|p| u.name(p.semantic.unwrap())).collect::<Vec<_>>(), ["SV_GroupID", "SV_GroupThreadID", "SV_GroupIndex"]);
    assert_eq!(main.line, 54);
    // every statement has a line, and they follow the source
    assert_eq!(u.stmts.len(), u.stmt_lines.len());
    let body = u.block(main.body.unwrap());
    assert!(body.windows(2).all(|w| u.stmt_lines[w[0].0 as usize] <= u.stmt_lines[w[1].0 as usize]));
    let (loop_id, loop_stmt) = body.iter().map(|&s| (s, u.stmt(s))).find(|(_, s)| matches!(s, Stmt::While { .. })).unwrap();
    assert_eq!(u.name(u.stmt_attrs[&loop_id.0][0].name), "loop");
    assert_eq!(u.stmt_lines[loop_id.0 as usize], 64);
    let Stmt::While { body, .. } = *loop_stmt else { unreachable!() };
    assert!(body.0 < loop_id.0, "children come before their statement");
    // one TypeId per type
    let uints = u.types.iter().filter(|t| matches!(t, TypeRef::Num { kind: xcl_dxil::hlsl::ast::ScalarKind::Uint, rows: 1, cols: 1 })).count();
    assert_eq!(uints, 1);
    assert!(u.globals.iter().any(|g| matches!(g, Global::Struct(_))));
    assert!(u.exprs.iter().any(|e| matches!(e, Expr::Float(v, _) if *v == 1e-8f32 as f64)));
}

/// No input makes the parser panic: every prefix of a program, and the program with one
/// byte replaced, is either parsed or refused with an error that has a line in the source.
#[test]
fn damaged_programs_are_errors_not_panics() {
    let check = |text: &str| match parse::parse(text) {
        Ok(u) => {
            // whatever parses also prints and parses again
            let printed = astprint::unit(&u);
            if let Err(e) = parse::parse(&printed) {
                panic!("the printed form of an accepted text does not parse: {e}\n--- text\n{text}\n--- printed\n{printed}");
            }
        }
        Err(Error::Source { line, .. }) => assert!(line >= 1 && line as usize <= text.lines().count() + 1, "line {line} of {}", text.lines().count()),
        Err(Error::Unsupported(what)) => assert!(!what.is_empty()),
        Err(Error::Internal(e)) => panic!("internal error: {e}\n{text}"),
    };
    for (_, src) in SAMPLES {
        for end in 0..src.len() {
            if src.is_char_boundary(end) {
                check(&src[..end]);
            }
        }
    }
    let replacements = b"(){}[];,#\"'\\/.*0aZ<>=?:+-!~&|^% \n\t9_u";
    for (k, (_, src)) in SAMPLES.iter().enumerate() {
        let mut bytes = src.as_bytes().to_vec();
        for at in (k % 3..bytes.len()).step_by(3) {
            let old = bytes[at];
            for (n, &r) in replacements.iter().enumerate() {
                if (at + n) % 4 != 0 {
                    continue;
                }
                bytes[at] = r;
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    check(text);
                }
            }
            bytes[at] = old;
        }
    }
    // bytes that are not text at all
    let mut seed = 0x2545F4914F6CDD1Du64;
    for _ in 0..2000 {
        let text: String = (0..64)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                char::from_u32((seed >> 32) as u32 % 0x250).unwrap_or(' ')
            })
            .collect();
        check(&text);
    }
}

/// The deepest nesting the parser accepts fits a small stack, and one level more is refused
/// (not a stack overflow), here and in the printers that walk the tree.
#[test]
fn nesting_is_bounded() {
    let run = |src: String| {
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || parse::parse(&src).map(|u| (astprint::unit(&u).len(), astprint::tree(&u).len())))
            .unwrap()
            .join()
            .expect("no panic, no overflow")
    };
    let parens = |n: usize| format!("void f() {{ a = {}b{}; }}", "(".repeat(n), ")".repeat(n));
    assert!(run(parens(parse::MAX_DEPTH as usize - 8)).is_ok());
    assert_eq!(run(parens(parse::MAX_DEPTH as usize + 1)), Err(Error::Unsupported(format!("nesting deeper than {} levels", parse::MAX_DEPTH))));
    let blocks = |n: usize| format!("void f() {{ {}a = b;{} }}", "{".repeat(n), "}".repeat(n));
    assert!(run(blocks(parse::MAX_DEPTH as usize - 8)).is_ok());
    assert!(matches!(run(blocks(parse::MAX_DEPTH as usize + 1)), Err(Error::Unsupported(_))));
    let ifs = |n: usize| format!("void f() {{ {} a = b; }}", "if (c) a = 1; else ".repeat(n));
    assert!(run(ifs(parse::MAX_DEPTH as usize / 2)).is_ok());
    assert!(matches!(run(ifs(parse::MAX_DEPTH as usize + 1)), Err(Error::Unsupported(_))));
    let unary = |n: usize| format!("void f() {{ a = {}b; }}", "-(uint)~".repeat(n));
    assert!(run(unary(100)).is_ok());
    assert!(matches!(run(unary(parse::MAX_DEPTH as usize)), Err(Error::Unsupported(_))));
    // trees that are deep without any nesting in the text
    let tall = Err(Error::Unsupported(format!("expression tree deeper than {} nodes", parse::MAX_HEIGHT)));
    let sum = |n: usize| format!("void f() {{ a = b{}; }}", " + b".repeat(n));
    assert!(run(sum(parse::MAX_HEIGHT as usize - 8)).is_ok());
    assert_eq!(run(sum(parse::MAX_HEIGHT as usize + 1)), tall);
    let members = |n: usize| format!("void f() {{ a = b{}; }}", ".x[0]".repeat(n));
    assert!(run(members(parse::MAX_HEIGHT as usize / 2 - 8)).is_ok());
    assert_eq!(run(members(parse::MAX_HEIGHT as usize)), tall);
    let commas = |n: usize| format!("void f() {{ a = (b{}); }}", ", b".repeat(n));
    assert_eq!(run(commas(parse::MAX_HEIGHT as usize + 1)), tall);
    // lists are not nesting: a long argument list or initialiser is fine
    let list = format!("static const uint T[5000] = {{ 0u{} }};", ", 1u".repeat(4999));
    assert!(run(list).is_ok());
}

// ---- the translator's preludes ----

fn emit_rs() -> Option<String> {
    let path = match std::env::var_os("XCL_DXIL_EMIT_RS") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../xcl/src/gpu/recomp/emit.rs"),
    };
    let text = std::fs::read_to_string(&path).ok();
    if text.is_none() {
        eprintln!("{} is not there: skipped", path.display());
    }
    text
}

/// The text of `const NAME: &str = r#"..."#;` in the translator.
fn prelude<'a>(emit: &'a str, name: &str) -> &'a str {
    let head = format!("const {name}: &str = r#\"");
    let start = emit.find(&head).unwrap_or_else(|| panic!("no {name} in emit.rs")) + head.len();
    let end = start + emit[start..].find("\"#;").unwrap();
    &emit[start..end]
}

/// Declarations of the constant tables that `text` reads and the translator's code writes
/// (`static const T NAME[n] = ...` in a format string of emit.rs, not in a prelude): their
/// values are computed there, and any will do here.
fn tables(emit: &str, text: &str) -> String {
    let mut out = String::new();
    for (at, _) in emit.match_indices("\"static const ") {
        let head = emit[at + 1..].split(" = ").next().unwrap_or("");
        let mut words = head.split_whitespace().skip(2);
        let (Some(ty), Some(array)) = (words.next(), words.next()) else { continue };
        let Some((name, len)) = array.strip_suffix(']').and_then(|a| a.split_once('[')) else { continue };
        if !text.contains(&format!("{name}[")) || out.contains(&format!(" {name}[")) {
            continue;
        }
        // a length the translator computes (`[{}]`): one that the prelude's indexing fits in
        let n = len.parse().unwrap_or(1024);
        let one = match ty.chars().last().and_then(|c| c.to_digit(10)) {
            Some(k) => format!("{ty}({})", vec!["0x00082001u"; k as usize].join(", ")),
            None => "0x00082001u".to_string(),
        };
        out.push_str(&format!("static const {ty} {name}[{n}] = {{ {} }};\n", vec![one; n].join(", ")));
    }
    out
}

/// The preludes as the translator combines them, for each choice of the macros they test.
fn prelude_units(emit: &str) -> Vec<(String, String)> {
    let mem = [prelude(emit, "PRELUDE_MEM"), prelude(emit, "PRELUDE_MEM_ACCESS"), prelude(emit, "PRELUDE_SCRATCH")].concat();
    let table = tables(emit, &mem);
    let mut units = Vec::new();
    for pixel in [false, true] {
        let mut s = String::from("static uint lane_id;\nvoid xcl_trip(uint k) {}\nRWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\nRWByteAddressBuffer mem : register(u2);\n");
        if pixel {
            s.insert_str(0, "#define XCL_PIXEL 1\n");
        }
        s.push_str("#define XCL_LOOP_GUARD 65536u\n");
        s.push_str(prelude(emit, "PRELUDE"));
        s.push_str(&table);
        s.push_str(&mem);
        units.push((format!("prelude, pixel {pixel}"), s));
    }
    for (spin, pair) in [(0, 0), (0, 1), (1, 1)] {
        let mut s = String::from("static uint lane_id;\nvoid xcl_trip(uint k) {}\nstatic uint xcl_half = 0u;\ngroupshared uint xcl_lds[8192];\n");
        s.push_str(&format!("#define XCL_X_THREADS 96u\n#define XCL_X_WAVES 2u\n#define XCL_X_SLOT 64u\n#define XCL_X_SPIN {spin}\n#define XCL_X_PAIR {pair}\n"));
        s.push_str("#define XCL_X_LDS_TOP 7000u\n#define XCL_XS(i) xcl_lds[7001u + (i)]\n#define XCL_XC(i) xcl_lds[8100u + (i)]\n#define XCL_XW xcl_lds[8191u]\n");
        s.push_str("#define XCL_BAR_ALIVE xcl_lds[8189u]\n#define XCL_BAR_STATE xcl_lds[8190u]\n");
        s.push_str("#define XCL_PROG xcl_lds[8188u]\n");
        s.push_str(prelude(emit, "PRELUDE_TICK"));
        s.push_str(prelude(emit, "PRELUDE_X64"));
        s.push_str(prelude(emit, if spin == 1 { "PRELUDE_X64_MEET_COUNTED" } else { "PRELUDE_X64_MEET" }));
        s.push_str(prelude(emit, "PRELUDE_X64_LANES"));
        s.push_str(prelude(emit, "PRELUDE_SAFE_BARRIER"));
        units.push((format!("x64 prelude, spin {spin}, pair {pair}"), s));
    }
    units
}

#[test]
fn translator_preludes_round_trip() {
    let Some(emit) = emit_rs() else { return };
    for (what, src) in prelude_units(&emit) {
        round_trip(&what, &src);
        let u = parse::parse(&src).unwrap();
        assert!(u.funcs.len() >= 8, "{what}: {} functions", u.funcs.len());
    }
}

// ---- DXC as the oracle ----

/// DXC's disassembly (`-Fc`) of `src` at `-O1` without its comment lines (they carry the
/// hash), or its error messages.
fn dxc(dir: &Path, src: &str, profile: &str) -> Result<String, String> {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = std::env::temp_dir().join(format!("xcl-dxil-parser-{}-{n}", std::process::id()));
    let (input, output) = (stem.with_extension("hlsl"), stem.with_extension("ll"));
    std::fs::write(&input, src).unwrap();
    let mut cmd = Command::new(dir.join("bin/dxc"));
    cmd.env("LD_LIBRARY_PATH", dir.join("lib")).args(["-T", profile, "-HV", "2021", "-Gis", "-O1", "-Fc"]).arg(&output).arg(&input);
    if profile.starts_with("lib") {
        // keep every function, not only what an entry point reaches
        cmd.args(["-default-linkage", "external"]);
    } else {
        cmd.args(["-E", "main"]);
    }
    let out = cmd.output().expect("dxc runs");
    let text = std::fs::read_to_string(&output);
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    match text {
        Ok(t) if out.status.success() => Ok(t.lines().filter(|l| !l.starts_with(';')).collect::<Vec<_>>().join("\n")),
        _ => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
    }
}

fn dxc_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("XCL_DXIL_DXC").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("XCL_DXIL_DXC is not set: skipped");
    }
    dir
}

/// The tree means what the source meant: DXC compiles the printed text to the same DXIL.
fn same_dxil(dir: &Path, what: &str, src: &str, profile: &str) {
    let printed = round_trip(what, src);
    let a = dxc(dir, src, profile).unwrap_or_else(|e| panic!("{what}: DXC rejects the source:\n{e}"));
    let b = dxc(dir, &printed, profile).unwrap_or_else(|e| panic!("{what}: DXC rejects the printed text:\n{e}\n{printed}"));
    assert!(a.lines().filter(|l| l.starts_with("  ")).count() > 20, "{what}: nothing was compiled");
    assert!(a == b, "{what}: DXC compiles the printed text to other DXIL");
}

#[test]
fn dxc_compiles_the_printed_samples_to_the_same_dxil() {
    let Some(dir) = dxc_dir() else { return };
    for (profile, src) in SAMPLES {
        same_dxil(&dir, profile, src, profile);
    }
}

/// Every function of the preludes, as a library (an entry point would leave DXC only the
/// functions it calls).
#[test]
fn dxc_compiles_the_printed_preludes_to_the_same_dxil() {
    let (Some(dir), Some(emit)) = (dxc_dir(), emit_rs()) else { return };
    for (what, src) in prelude_units(&emit) {
        same_dxil(&dir, &what, &src, "lib_6_6");
    }
}

/// What the tree says about literals is what DXC does with them.
#[test]
fn literals_mean_what_dxc_says() {
    let Some(dir) = dxc_dir() else { return };
    let value = |literal: &str| {
        let src = format!("RWByteAddressBuffer io : register(u0);\n[numthreads(1, 1, 1)] void main() {{ io.Store(0u, {literal}); }}\n");
        let ll = dxc(&dir, &src, "cs_6_6").unwrap_or_else(|e| panic!("{literal}: {e}"));
        let store = ll.lines().find(|l| l.contains("@dx.op.rawBufferStore.i32(i32 140")).unwrap().to_string();
        let v: i64 = store.split("i32 ").nth(4).unwrap().trim_end_matches([',', ' ']).parse().unwrap();
        v as u32
    };
    let parsed = |literal: &str| {
        let u = parse::parse(&format!("void f() {{ a = {literal}; }}")).unwrap();
        *u.exprs.iter().find(|e| matches!(e, Expr::Float(..) | Expr::Int(..))).unwrap()
    };
    // `f`: one rounding, to float; no suffix: to double first
    for text in ["1.00000005960464477539062500000000001", "0.1", "16777217.0", "3.4028235677973366e38", "1e-45", "6.2831853"] {
        for suffix in ["f", ""] {
            let literal = format!("{text}{suffix}");
            let Expr::Float(v, _) = parsed(&literal) else { panic!() };
            assert_eq!((v as f32).to_bits(), value(&format!("asuint({literal})")), "{literal}");
        }
    }
    // `ull` / `ll` are 64-bit; `ul` / `l` / `u` are not
    use xcl_dxil::hlsl::ast::IntSuffix;
    for (suffix, kind, wide) in [("ull", IntSuffix::ULL, true), ("ll", IntSuffix::LL, true), ("ul", IntSuffix::UL, false), ("l", IntSuffix::L, false), ("u", IntSuffix::U, false)] {
        assert_eq!(parsed(&format!("3{suffix}")), Expr::Int(3, kind));
        let hi = value(&format!("(uint)((uint64_t)(0xFFFFFFFF{suffix} + 0xFFFFFFFF{suffix}) >> 32)"));
        assert_eq!(hi != 0, wide, "{suffix}: {hi}");
    }
    // octal, hexadecimal
    assert_eq!(parsed("017"), Expr::Int(15, IntSuffix::None));
    assert_eq!(value("017"), 15);
    assert_eq!(value("0x1Fu"), 31);
}

/// What DXC lets a declaration name, the parser does too (and what it reserves is refused).
#[test]
fn names_dxc_accepts_are_accepted() {
    let Some(dir) = dxc_dir() else { return };
    let words = [
        "half", "uint", "uint4", "float4x4", "dword", "min16float", "vector", "matrix", "Texture2D", "SamplerState", "uint64_t", "string", "sample",
        "vertices", "indices", "primitives", "payload", "precise", "globallycoherent", "float1", "min10float", "float", "int", "bool", "double", "void",
        "line", "point", "triangle", "linear", "centroid", "static", "const", "in", "out", "inout", "register", "nointerpolation", "uniform", "struct",
        "switch", "default", "true", "discard", "typedef", "cbuffer", "volatile", "inline", "groupshared", "row_major",
    ];
    for w in words {
        let src = format!(
            "RWByteAddressBuffer io : register(u0);\nuint twice(uint {w}) {{ return {w} * 2u; }}\n[numthreads(1, 1, 1)] void main() {{ uint q = io.Load(0u); {{ uint {w} = q + 1u; q = ({w}) - twice({w}); }} uint4 r = uint4(q, q, q, q); io.Store(4u, r.x); }}\n"
        );
        match (dxc(&dir, &src, "cs_6_6"), parse::parse(&src)) {
            (Ok(a), Ok(u)) => {
                let b = dxc(&dir, &astprint::unit(&u), "cs_6_6").unwrap_or_else(|e| panic!("{w}: {e}"));
                assert!(a == b, "{w}: other DXIL for the printed text");
            }
            (Err(_), Err(_)) => {}
            (Ok(_), Err(e)) => panic!("DXC accepts a variable named {w}, the parser does not: {e}"),
            (Err(e), Ok(_)) => panic!("the parser accepts a variable named {w}, DXC does not: {e}"),
        }
    }
}
