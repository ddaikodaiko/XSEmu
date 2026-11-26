use std::num::NonZeroU32;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use windows::core::{s, HRESULT, PCWSTR};
use windows::Win32::Foundation::HWND;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS};

type CaptureNextFrame = unsafe extern "system" fn(PCWSTR, u32) -> HRESULT;
type SetGlobalTargetWindow = unsafe extern "system" fn(HWND);

struct Capture {
    at_s: f64,
    frames: NonZeroU32,
    path: PathBuf,
    set_window: SetGlobalTargetWindow,
    next_frame: CaptureNextFrame,
    requested: AtomicBool,
}

static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();

pub(crate) fn load_before_d3d12() {
    CAPTURE.get_or_init(|| {
        if !std::env::var_os("XCL_PIX_CAPTURE").is_some_and(|value| value == "1") {
            return None;
        }
        match load() {
            Ok(capture) => Some(capture),
            Err(error) => {
                crate::xlog_warn!("PIX: GPU capture disabled: {error}");
                None
            }
        }
    });
}

fn load() -> Result<Capture, String> {
    let at_s: f64 = std::env::var("XCL_PIX_CAPTURE_AT_S")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|value: &f64| value.is_finite() && *value >= 0.0)
        .ok_or("XCL_PIX_CAPTURE_AT_S must be a finite, nonnegative number")?;
    let frames = match std::env::var("XCL_PIX_CAPTURE_FRAMES") {
        Ok(value) => value
            .parse::<NonZeroU32>()
            .map_err(|_| "XCL_PIX_CAPTURE_FRAMES must be a positive integer")?,
        Err(std::env::VarError::NotPresent) => NonZeroU32::MIN,
        Err(_) => return Err("XCL_PIX_CAPTURE_FRAMES must be a positive integer".into()),
    };
    let tag = match std::env::var("XCL_PIX_CAPTURE_TAG") {
        Ok(tag) if !tag.is_empty() && tag.len() <= 64 && tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') => tag,
        Err(std::env::VarError::NotPresent) => format!("xcl-{}", std::process::id()),
        _ => return Err("XCL_PIX_CAPTURE_TAG must be 1–64 ASCII letters, digits, hyphens, or underscores".into()),
    };
    let dll_name = "WinPixGpuCapturer.dll";
    let root = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .ok_or("ProgramFiles is not an absolute path")?
        .join("Microsoft PIX");
    let missing_dll = || format!("{dll_name} not found under {}; install Microsoft PIX", root.display());
    let dll = std::fs::read_dir(&root)
        .map_err(|_| missing_dll())?
        .filter_map(Result::ok)
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .map(|entry| entry.path().join(dll_name))
        .filter(|path| path.is_file())
        .max()
        .ok_or_else(missing_dll)?;

    let dir = Path::new(r"D:\xsemu-capy-work-archive\pix");
    std::fs::create_dir_all(dir).map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let path = dir.join(format!("{tag}.wpix"));
    if path.exists() {
        return Err(format!("{} already exists; choose a new XCL_PIX_CAPTURE_TAG", path.display()));
    }

    let wide: Vec<u16> = dll.as_os_str().encode_wide().chain([0]).collect();
    let module = unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
        .map_err(|error| format!("cannot load {}: {error}", dll.display()))?;
    let (set_window, next_frame) = unsafe {
        let set_window = GetProcAddress(module, s!("SetGlobalTargetWindow")).ok_or("PIXSetTargetWindow export unavailable")?;
        let next_frame = GetProcAddress(module, s!("CaptureNextFrame")).ok_or("PIXGpuCaptureNextFrames export unavailable")?;
        (
            std::mem::transmute::<_, SetGlobalTargetWindow>(set_window),
            std::mem::transmute::<_, CaptureNextFrame>(next_frame),
        )
    };
    crate::xlog_info!(
        "PIX: loaded {}; will request {frames} frame(s) at {at_s:.3} s to {}",
        dll.display(),
        path.display()
    );
    Ok(Capture {
        at_s,
        frames,
        path,
        set_window,
        next_frame,
        requested: AtomicBool::new(false),
    })
}

pub(crate) fn before_present(hwnd: HWND) {
    let Some(Some(capture)) = CAPTURE.get() else { return };
    if capture.requested.load(Ordering::Relaxed) {
        return;
    }
    let now = crate::log::uptime();
    if now < capture.at_s || capture.requested.swap(true, Ordering::Relaxed) {
        return;
    }
    let wide: Vec<u16> = capture.path.as_os_str().encode_wide().chain([0]).collect();
    let result = unsafe {
        (capture.set_window)(hwnd);
        (capture.next_frame)(PCWSTR(wide.as_ptr()), capture.frames.get())
    };
    match result.ok() {
        Ok(()) => crate::xlog_info!("PIX: queued {} GPU frame(s) at {now:.3} s", capture.frames),
        Err(error) => crate::xlog_warn!("PIX: GPU capture request failed at {now:.3} s: {error}"),
    }
}
