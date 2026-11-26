use std::path::Path;

// The translated-shader cache is keyed on the console shader bytes, the DXC toolchain and the
// translator's code: the sources below with comments, whitespace, log lines and test code
// stripped, so any code change re-keys it and a log-only or comment-only edit keeps it warm.
// Bump this only to force a flush.
const TRANSLATOR_FLUSH_SALT: &str = "1";

const TRANSLATOR_DIRS: &[&str] = &["src/gpu/recomp"];
const TRANSLATOR_FILES: &[&str] = &[
    "src/gpu/dxbc.rs",
    "src/gpu/pack.rs",
    "src/gpu/rdna2.rs",
    "src/gpu/rdna2_spec.rs",
    "src/gpu/rdna2_xbox.rs",
    "src/gpu/shader.rs",
    "src/gpu/xbox.rs",
    "src/shaders.rs",
];
const LOG_MACROS: &[&[u8]] = &[b"println", b"eprintln", b"print", b"eprint"];

fn add(h: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *h ^= byte as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whitespace or a comment at `i`: where it ends.
fn trivia_end(b: &[u8], i: usize) -> Option<usize> {
    if b[i].is_ascii_whitespace() {
        return Some(i + b[i..].iter().take_while(|c| c.is_ascii_whitespace()).count());
    }
    if b[i..].starts_with(b"//") {
        return Some(b[i..].iter().position(|&c| c == b'\n').map_or(b.len(), |n| i + n));
    }
    if b[i..].starts_with(b"/*") {
        let (mut j, mut depth) = (i + 2, 1);
        while j < b.len() && depth > 0 {
            if b[j..].starts_with(b"/*") {
                depth += 1;
                j += 2;
            } else if b[j..].starts_with(b"*/") {
                depth -= 1;
                j += 2;
            } else {
                j += 1;
            }
        }
        return Some(j);
    }
    None
}

/// A string, raw string or char literal at `i`: where it ends.
fn literal_end(b: &[u8], i: usize) -> Option<usize> {
    let prefixed = i > 0 && is_ident(b[i - 1]);
    let mut j = i;
    if !prefixed && b[j] == b'b' && j + 1 < b.len() && matches!(b[j + 1], b'"' | b'r' | b'\'') {
        j += 1;
    }
    if (!prefixed || j > i) && b[j] == b'r' {
        let hashes = b[j + 1..].iter().take_while(|&&c| c == b'#').count();
        if b.get(j + 1 + hashes) == Some(&b'"') {
            let mut close = vec![b'"'];
            close.extend(std::iter::repeat_n(b'#', hashes));
            let body = j + 2 + hashes;
            return Some(b[body..].windows(close.len()).position(|w| w == close.as_slice()).map_or(b.len(), |n| body + n + close.len()));
        }
        return None;
    }
    if b[j] == b'"' && (!prefixed || j > i) {
        let mut k = j + 1;
        while k < b.len() && b[k] != b'"' {
            k += if b[k] == b'\\' { 2 } else { 1 };
        }
        return Some((k + 1).min(b.len()));
    }
    if b[j] == b'\'' && (!prefixed || j > i) {
        if b.get(j + 1) == Some(&b'\\') {
            return b[j + 2..].iter().position(|&c| c == b'\'').map(|n| j + 3 + n);
        }
        let len = std::str::from_utf8(&b[j + 1..(j + 5).min(b.len())]).map_or_else(|e| e.valid_up_to(), |s| s.len());
        let ch = std::str::from_utf8(&b[j + 1..j + 1 + len]).ok()?.chars().next()?.len_utf8();
        return (b.get(j + 1 + ch) == Some(&b'\'')).then_some(j + 2 + ch);
    }
    None
}

/// From the opening delimiter at `i`: just past its match.
fn balanced_end(b: &[u8], mut i: usize) -> usize {
    let mut depth = 0usize;
    while i < b.len() {
        if let Some(end) = trivia_end(b, i).or_else(|| literal_end(b, i)) {
            i = end;
            continue;
        }
        match b[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return i + 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

fn skip_ws(b: &[u8], i: usize) -> usize {
    i + b[i..].iter().take_while(|c| c.is_ascii_whitespace()).count()
}

/// A log macro invocation or a `#[cfg(test)]` item at `i`: where it ends.
fn ignored_end(b: &[u8], i: usize) -> Option<usize> {
    if b[i..].starts_with(b"#[cfg(test)]") {
        let mut j = i + b"#[cfg(test)]".len();
        while j < b.len() {
            if let Some(end) = trivia_end(b, j).or_else(|| literal_end(b, j)) {
                j = end;
                continue;
            }
            match b[j] {
                b';' => return Some(j + 1),
                b'{' => return Some(balanced_end(b, j)),
                b'(' | b'[' => j = balanced_end(b, j),
                _ => j += 1,
            }
        }
        return Some(b.len());
    }
    if i > 0 && is_ident(b[i - 1]) {
        return None;
    }
    let name = b[i..].iter().take_while(|&&c| is_ident(c)).count();
    let word = &b[i..i + name];
    if name == 0 || !(word.starts_with(b"xlog_") || LOG_MACROS.contains(&word)) {
        return None;
    }
    let bang = skip_ws(b, i + name);
    if b.get(bang) != Some(&b'!') {
        return None;
    }
    let open = skip_ws(b, bang + 1);
    if !matches!(b.get(open), Some(b'(' | b'[' | b'{')) {
        return None;
    }
    let end = balanced_end(b, open);
    let semi = skip_ws(b, end);
    Some(if b.get(semi) == Some(&b';') { semi + 1 } else { end })
}

/// The translator's code as the compiler sees it, minus what cannot change translated output.
fn normalized(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        if let Some(end) = trivia_end(src, i) {
            i = end;
        } else if let Some(end) = ignored_end(src, i) {
            while out.ends_with(b"::") {
                out.truncate(out.len() - 2);
                while out.last().is_some_and(|&c| is_ident(c)) {
                    out.pop();
                }
            }
            i = end;
        } else if let Some(end) = literal_end(src, i) {
            out.extend_from_slice(&src[i..end]);
            i = end;
        } else {
            out.push(src[i]);
            i += 1;
        }
    }
    out
}

/// A compiled resource file (.res, what rc.exe makes of `1 ICON "XSEmu.ico"`) holding the icon
/// `ico`: each of its images as an RT_ICON (ids 1, 2, …) and the RT_GROUP_ICON 1 that lists
/// them. The linker (link.exe, lld-link) takes a .res as an input.
fn icon_res(ico: &[u8]) -> Vec<u8> {
    let u16_at = |at: usize| u16::from_le_bytes([ico[at], ico[at + 1]]);
    let u32_at = |at: usize| u32::from_le_bytes([ico[at], ico[at + 1], ico[at + 2], ico[at + 3]]);
    let entry = |res: &mut Vec<u8>, kind: u16, id: u16, flags: u16, language: u16, data: &[u8]| {
        res.extend_from_slice(&(data.len() as u32).to_le_bytes());
        res.extend_from_slice(&32u32.to_le_bytes());
        for v in [0xFFFF, kind, 0xFFFF, id] {
            res.extend_from_slice(&u16::to_le_bytes(v));
        }
        res.extend_from_slice(&0u32.to_le_bytes());
        res.extend_from_slice(&flags.to_le_bytes());
        res.extend_from_slice(&language.to_le_bytes());
        res.extend_from_slice(&[0; 8]);
        res.extend_from_slice(data);
        res.resize(res.len().next_multiple_of(4), 0);
    };
    const RT_ICON: u16 = 3;
    const RT_GROUP_ICON: u16 = 14;
    const EN_US: u16 = 0x0409;
    let mut res = Vec::new();
    entry(&mut res, 0, 0, 0, 0, &[]);
    let count = u16_at(4);
    let mut group = vec![0, 0, 1, 0];
    group.extend_from_slice(&count.to_le_bytes());
    for k in 0..count as usize {
        let at = 6 + 16 * k;
        let (len, offset) = (u32_at(at + 8) as usize, u32_at(at + 12) as usize);
        entry(&mut res, RT_ICON, k as u16 + 1, 0x1010, EN_US, &ico[offset..offset + len]);
        group.extend_from_slice(&ico[at..at + 4]);
        group.extend_from_slice(&u16_at(at + 4).max(1).to_le_bytes());
        group.extend_from_slice(&ico[at + 6..at + 12]);
        group.extend_from_slice(&(k as u16 + 1).to_le_bytes());
    }
    entry(&mut res, RT_GROUP_ICON, 1, 0x1030, EN_US, &group);
    res
}

/// The exact hooks below only affect the uncached material-gate variant. Changes to other
/// translator code, including edits to the hook lines themselves, still rekey the cache.
fn translator_source(relative: &str, src: &[u8]) -> Vec<u8> {
    let hooks: &[&[u8]] = match relative {
        "src/gpu/recomp/pso.rs" => &[b"pub mod material_gate;"],
        "src/gpu/recomp/emit.rs" => &[
            b"    material_gate: Option<std::rc::Rc<super::pso::material_gate::Trace>>,",
            b"            material_gate: super::pso::material_gate::current(),",
            b"        if let Some(line) = self.material_gate.as_ref().and_then(|gate| gate.before(d, self.packed_compute())) { self.line(&line); }",
            b"                if let Some(line) = self.material_gate.as_ref().and_then(|gate| gate.store(self.cur_pc, dim, &tw0, &icoord, act)) { self.line(&line); }",
            b"                if self.material_gate.is_some() { h.push_str(super::pso::material_gate::ENTRY); }",
        ],
        _ => return normalized(src),
    };
    let mut ordinary = Vec::with_capacity(src.len());
    for line in src.split_inclusive(|&byte| byte == b'\n') {
        let code = line.strip_suffix(b"\n").unwrap_or(line);
        let code = code.strip_suffix(b"\r").unwrap_or(code);
        if !hooks.contains(&code) {
            ordinary.extend_from_slice(line);
        }
    }
    normalized(&ordinary)
}

/// xcl-dxil's `BUILD_ID` as its build.rs computes it: FNV-1a 64 over every file under src, by
/// name with '/' in byte order, each name and text (without carriage returns) followed by a 0.
fn dxil_build_id(crate_root: &Path) -> String {
    fn collect(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read xcl-dxil sources").flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else {
                out.push(p);
            }
        }
    }
    let src = crate_root.join("src");
    println!("cargo:rerun-if-changed={}", src.display());
    let mut files = Vec::new();
    collect(&src, &mut files);
    let mut named: Vec<(String, std::path::PathBuf)> = files.into_iter().map(|p| (p.strip_prefix(crate_root).unwrap_or(&p).to_string_lossy().replace('\\', "/"), p)).collect();
    named.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for (name, path) in &named {
        add(&mut h, name.as_bytes());
        add(&mut h, &[0]);
        let text: Vec<u8> = std::fs::read(path).expect("read xcl-dxil source").into_iter().filter(|&b| b != b'\r').collect();
        add(&mut h, &text);
        add(&mut h, &[0]);
    }
    format!("{h:016x}")
}
fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources: Vec<String> = TRANSLATOR_FILES.iter().map(|s| s.to_string()).collect();
    for dir in TRANSLATOR_DIRS {
        println!("cargo:rerun-if-changed={dir}");
        for entry in std::fs::read_dir(root.join(dir)).expect("read shader translator directory") {
            let name = entry.expect("read shader translator directory").file_name().into_string().expect("translator source name");
            if name.ends_with(".rs") && name != "tests.rs" {
                sources.push(format!("{dir}/{name}"));
            }
        }
    }
    sources.sort();
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    add(&mut h, b"xcl-translator-normalized-v1\0");
    add(&mut h, TRANSLATOR_FLUSH_SALT.as_bytes());
    add(&mut h, &[0]);
    for relative in &sources {
        println!("cargo:rerun-if-changed={relative}");
        add(&mut h, relative.as_bytes());
        add(&mut h, &[0]);
        add(&mut h, &translator_source(relative, &std::fs::read(root.join(relative)).expect("read shader translator source")));
        add(&mut h, &[0]);
    }
    // the direct DXIL backend writes the containers the translated shaders become: its build id
    // (xcl-dxil's build.rs, computed the same way here) is part of the fingerprint, so a backend
    // change rekeys it like a translator change and the warm-up runs
    let dxil = dxil_build_id(&root.join("../xcl-dxil"));
    println!("cargo:rustc-env=XCL_DXIL_SOURCE_ID={dxil}");
    add(&mut h, b"xcl-dxil\0");
    add(&mut h, dxil.as_bytes());
    add(&mut h, &[0]);
    println!("cargo:rustc-env=XCL_TRANSLATOR_SOURCE_ID={h:016x}");

    // the segment heap: the NT heap's one backend lock held a dozen threads compiling in the
    // GPU driver at a tenth of a core each
    println!("cargo:rerun-if-changed=xcl.manifest");
    if std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|e| e == "msvc") {
        let manifest = root.join("xcl.manifest");
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:{}", manifest.display());
    }

    // XSEmu.exe's icon (assets/logo/XSEmu.ico)
    println!("cargo:rerun-if-changed=../../assets/logo/XSEmu.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows") && std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|e| e == "msvc") {
        let ico = std::fs::read(root.join("../../assets/logo/XSEmu.ico")).expect("read assets/logo/XSEmu.ico");
        let res = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join("XSEmu.res");
        std::fs::write(&res, icon_res(&ico)).expect("write XSEmu.res");
        println!("cargo:rustc-link-arg-bins={}", res.display());
    }
}
