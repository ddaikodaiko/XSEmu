use crate::gpu::rdna2;
use crate::gpu::recomp::{DecodedProgram, Mode, Options};

pub(super) struct Site {
    pub(super) after: u64,
    source: u64,
    uv: [u64; 2],
    data: u64,
    threads: [u32; 3],
}

impl Site {
    pub(super) fn new(pc: u64, program: &DecodedProgram<'_>, opt: &Options) -> Result<Self, String> {
        let Mode::Compute(cs) = &opt.mode else { return Err("source-depth requires compute".into()) };
        let d = program.insts.iter().find(|d| d.pc == pc).ok_or("missing source-depth instruction")?;
        let field = |name: &str| rdna2::field_value(d.enc, &d.words, name).ok_or("missing MIMG field");
        let after = d.pc + d.len as u64;
        if d.inst.name != "IMAGE_GATHER4_LZ" || d.enc.name != "ENC_MIMG" || field("DIM")? != 1 || field("DMASK")? != 1
            || ["TFE", "LWE", "D16", "A16", "UNORM"].iter().any(|f| field(f) != Ok(0))
            || !program.starts.contains(&after) || program.leaders.contains(&after) || cs.threads[2] != 1 {
            return Err("source-depth needs a fall-through 2D red gather4_lz with 32-bit normal coordinates".into());
        }
        let w = program.static_samplers.get(&pc).ok_or("source-depth needs a proven static sampler")?;
        if ![w[0] & 7, (w[0] >> 3) & 7].into_iter().all(|c| matches!(c, 2 | 4)) {
            return Err("source-depth integer comparison requires clamp addressing".into());
        }
        let x = field("VADDR")?;
        let y = match field("NSA")? {
            0 => x + 1,
            n => rdna2::encoding(&format!("MIMG_NSA{n}")).and_then(|e| rdna2::field_value(e, &d.words, "VADDRA")).ok_or("unsupported Y coordinate")?,
        };
        let data = field("VDATA")?;
        let source = field("SRSRC")? * rdna2::register_scale("SRSRC");
        if [x, y].iter().any(|v| (data..data + 4).contains(v)) || source + 1 > 105 {
            return Err("source-depth post-gather operands are unavailable".into());
        }
        Ok(Self { after, source, uv: [x, y], data, threads: cs.threads })
    }

    pub(super) fn emit(&self, packed: bool, depth_plane: bool) -> String {
        let half = |p: &str| {
            let [x, y] = self.uv;
            let [tx, ty, _] = self.threads;
            let s = self.source;
            let t = format!("uint4({})", (self.data..self.data + 4).map(|r| format!("{p}v{r}")).collect::<Vec<_>>().join(", "));
            let hlsl = r#"
if (xcl_gate_on && @p@xcl_on) {
    uint ix = @s@;
    float2 uv = asfloat(uint2(@x@, @y@));
    uint4 g = @t@;
    uint2 grid = 2u * (gid.xy * uint2(@tx@u, @ty@u) + @p@gtid.xy);
    uint2 size = uint2(res.Load(8u), res.Load(12u));
    bool match_index = ix == res.Load(68u) && ix != 0xffffffffu && ((@s1@ >> 20u) & 511u) == @tag@u;
    bool valid = false, covered = false, footprint = false, precise_match = false, same_base = false;
    bool finite_uv = all((asuint(uv) & 0x7fffffffu) < 0x7f800000u);
    bool unit_uv = finite_uv && all(uv >= 0.0f) && all(uv <= 1.0f);
    uint4 grid_load = 0u, uv_load = 0u;
    if (match_index) {
        Texture2D<float4> src = ResourceDescriptorHeap[NonUniformResourceIndex(ix)];
        uint w, h;
        src.GetDimensions(w, h);
        valid = w == size.x && h == size.y && w > 0u && h > 0u && w <= 16384u && h <= 16384u;
        if (valid) {
            covered = all(grid + 1u < size);
            if (covered) {
                int2 b = (int2)grid;
                grid_load = @gridload@;
            }
            if (unit_uv) {
                footprint = true;
                float2 pos = uv * float2(w, h) - 0.5f;
                int2 b = (int2)floor(pos);
                int2 lim = (int2)size - 1;
                uv_load = @uvload@;
                same_base = covered && all(b == (int2)grid);
                precise_match = all(b >= 0) && all(b + 1 < (int2)size) && all(frac(pos) > (1.0f / 256.0f)) && all(frac(pos) < (255.0f / 256.0f));
            }
            if (covered && all(grid == ((size / 2u) & ~1u))) {
                uint old; res.InterlockedAdd(136u, 1u, old);
                res.Store2(140u, asuint(uv)); res.Store4(148u, g); res.Store4(164u, grid_load); res.Store4(180u, uv_load);
            }
        }
    }
    @counts@
}"#;
            let loads = |clamp: bool| {
                let taps = [(0, 1), (1, 1), (1, 0), (0, 0)].map(|(dx, dy)| {
                    let addr = format!("b + int2({dx}, {dy})");
                    let addr = if clamp { format!("clamp({addr}, int2(0,0), lim)") } else { addr };
                    format!("asuint(src.Load(int3({addr}, 0)).r)")
                });
                format!("uint4({})", taps.join(", "))
            };
            let classified = |v, nan| {
                let bits = format!("({v} & 0x7fffffffu)");
                let test = if nan { format!("{bits} > 0x7f800000u") } else { format!("and({bits} != 0u, {bits} <= 0x7f800000u)") };
                format!("dot(uint4({test}), uint4(1u,1u,1u,1u))")
            };
            let counts = [
                "1u".into(), "(uint)match_index".into(), "(uint)valid".into(), "(uint)finite_uv".into(),
                "(uint)unit_uv".into(), "(uint)precise_match".into(), classified("g", false), classified("g", true),
                "4u * (uint)covered".into(), classified("grid_load", false), classified("grid_load", true),
                "4u * (uint)footprint".into(), classified("uv_load", false), classified("uv_load", true),
                "(uint)precise_match * dot(uint4(uv_load != g), uint4(1u,1u,1u,1u))".into(), "(uint)same_base".into(),
            ];
            let count_code = counts.iter().enumerate().map(|(i, val)| {
                format!("{{ uint n = WaveActiveSum({val}); if (WaveIsFirstLane() && n != 0u) {{ uint old; res.InterlockedAdd({}u, n, old); }} }}", (18 + i) * 4)
            }).collect::<Vec<_>>().join("\n");
            hlsl.replace("@tag@", if depth_plane { "0" } else { "22" }).replace("@p@", p).replace("@s@", &format!("s{s}")).replace("@s1@", &format!("s{}", s + 1))
                .replace("@x@", &format!("{p}v{x}")).replace("@y@", &format!("{p}v{y}")).replace("@t@", &t)
                .replace("@tx@", &tx.to_string()).replace("@ty@", &ty.to_string())
                .replace("@gridload@", &loads(false)).replace("@uvload@", &loads(true)).replace("@counts@", &count_code)
        };
        let mut text = half("");
        if packed { text.push_str(&half("hi_")); }
        text
    }
}
