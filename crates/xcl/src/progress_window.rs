//! A progress window and one updating console line for the long shader steps (`xcl warm-up
//! --blocking`, `xcl warm-up`): done of total with a bar, the rate
//! over the last minute, the time left, the RAM free and the giants compiling, and a Cancel
//! button where the step stops resumably (as Ctrl+C does). The console line is drawn only on a
//! terminal; a log keeps the step's own lines. (`xcl shaders --precompile` has none: shaders.rs
//! is a translator source, and any code change there would re-key the translated-shader cache.) `XCL_PROGRESS=0`: neither;
//! `XCL_PROGRESS_WINDOW=0`: no window.

use std::collections::VecDeque;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;

/// The rate is taken over this much of the recent past (the first keys of a list are its slowest).
const RATE_WINDOW: Duration = Duration::from_secs(60);
const TICK: Duration = Duration::from_millis(500);
const CANCEL_ID: usize = 1;
const WM_APP_CLOSE: u32 = WM_APP + 1;

struct Shared {
    step: &'static str,
    unit: &'static str,
    total: AtomicUsize,
    done: AtomicUsize,
    failed: AtomicUsize,
    cancel: Option<fn()>,
    cancelled: AtomicBool,
    ended: AtomicBool,
    /// (the window, 0 before it is made; the step ended): one lock, so the window is either made
    /// before the end and closed by it, or never shown
    window: Mutex<(isize, bool)>,
    /// (the status line, the bar's fraction)
    shown: Mutex<(String, f32)>,
    /// the length of the console line drawn last (0: none)
    drawn: Mutex<usize>,
    console: bool,
}

/// The window of the step running now (one at a time), for its window procedure.
static CURRENT: Mutex<Option<Arc<Shared>>> = Mutex::new(None);

/// The progress of one step, shown while it lives.
pub struct Monitor {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Monitor {
    /// `step` (as on the command line) over `total` `unit`s; `cancel` stops it resumably.
    pub fn start(step: &'static str, unit: &'static str, total: usize, cancel: Option<fn()>) -> Monitor {
        let on = crate::progress::enabled();
        let shared = Arc::new(Shared {
            step,
            unit,
            total: AtomicUsize::new(total),
            done: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            cancel,
            cancelled: AtomicBool::new(false),
            ended: AtomicBool::new(false),
            shown: Mutex::new((String::new(), 0.0)),
            drawn: Mutex::new(0),
            console: on && std::io::stderr().is_terminal(),
            window: Mutex::new((0, false)),
        });
        let mut threads = Vec::new();
        if !on {
            return Monitor { shared, threads };
        }
        let ticker = shared.clone();
        threads.extend(std::thread::Builder::new().name("xcl-progress".into()).spawn(move || tick(&ticker)).ok());
        if !std::env::var("XCL_PROGRESS_WINDOW").is_ok_and(|v| v.trim() == "0") {
            *CURRENT.lock().unwrap_or_else(|e| e.into_inner()) = Some(shared.clone());
            let window = shared.clone();
            threads.extend(std::thread::Builder::new().name("xcl-progress-window".into()).spawn(move || unsafe { run_window(&window) }).ok());
        }
        Monitor { shared, threads }
    }

    pub fn set_total(&self, total: usize) {
        self.shared.total.store(total, Ordering::Relaxed);
    }

    /// `done` of the total so far, `failed` of them failed.
    pub fn advance(&self, done: usize, failed: usize) {
        self.shared.done.store(done, Ordering::Relaxed);
        self.shared.failed.store(failed, Ordering::Relaxed);
    }

    /// The updating line is on the console: the step's periodic lines are left out.
    pub fn console(&self) -> bool {
        self.shared.console
    }

    /// Cancel was pressed.
    pub fn cancelled(&self) -> bool {
        self.shared.cancelled.load(Ordering::Relaxed)
    }

    /// A line of the step's own on stderr, above the updating line.
    pub fn println(&self, line: &str) {
        let mut drawn = self.shared.drawn.lock().unwrap_or_else(|e| e.into_inner());
        let mut err = std::io::stderr().lock();
        if *drawn != 0 {
            let _ = write!(err, "\r{:width$}\r", "", width = *drawn);
            *drawn = 0;
        }
        let _ = writeln!(err, "{line}");
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let s = &self.shared;
        s.ended.store(true, Ordering::Relaxed);
        let hwnd = {
            let mut window = s.window.lock().unwrap_or_else(|e| e.into_inner());
            window.1 = true;
            window.0
        };
        if hwnd != 0 {
            unsafe {
                let _ = PostMessageW(Some(HWND(hwnd as *mut _)), WM_APP_CLOSE, WPARAM(0), LPARAM(0));
            }
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        if s.console {
            refresh(s, &mut VecDeque::new(), Instant::now());
            let drawn = std::mem::take(&mut *s.drawn.lock().unwrap_or_else(|e| e.into_inner()));
            if drawn != 0 {
                eprintln!();
            }
        }
        let mut current = CURRENT.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|c| Arc::ptr_eq(c, s)) {
            *current = None;
        }
    }
}

fn tick(s: &Shared) {
    let mut samples = VecDeque::new();
    while !s.ended.load(Ordering::Relaxed) {
        refresh(s, &mut samples, Instant::now());
        std::thread::sleep(TICK);
    }
}

/// The status now into `s.shown`, and onto the console line.
fn refresh(s: &Shared, samples: &mut VecDeque<(Instant, usize)>, now: Instant) {
    let (done, total) = (s.done.load(Ordering::Relaxed), s.total.load(Ordering::Relaxed));
    samples.push_back((now, done));
    while samples.len() > 2 && samples.front().is_some_and(|&(at, _)| now.duration_since(at) > RATE_WINDOW) {
        samples.pop_front();
    }
    let p = Progress {
        step: s.step,
        unit: s.unit,
        done,
        total,
        failed: s.failed.load(Ordering::Relaxed),
        rate: rate(samples),
        free_mib: crate::hle::d3d12x::real::available_physical() >> 20,
        giants: crate::progress::LARGE_DRIVER.counts().pending,
        stopping: s.cancelled.load(Ordering::Relaxed),
    };
    let line = p.line();
    if s.console {
        let mut drawn = s.drawn.lock().unwrap_or_else(|e| e.into_inner());
        let width = line.chars().count();
        let _ = write!(std::io::stderr().lock(), "\r{line}{:pad$}", "", pad = drawn.saturating_sub(width));
        *drawn = width;
    }
    *s.shown.lock().unwrap_or_else(|e| e.into_inner()) = (line, p.fraction());
}

/// (done)/s between the oldest and newest sample, once they are 5 s apart.
fn rate(samples: &VecDeque<(Instant, usize)>) -> Option<f64> {
    let (&(t0, d0), &(t1, d1)) = (samples.front()?, samples.back()?);
    let dt = t1.duration_since(t0).as_secs_f64();
    (dt >= 5.0).then(|| d1.saturating_sub(d0) as f64 / dt)
}

struct Progress {
    step: &'static str,
    unit: &'static str,
    done: usize,
    total: usize,
    failed: usize,
    rate: Option<f64>,
    free_mib: u64,
    giants: usize,
    stopping: bool,
}

impl Progress {
    fn fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { (self.done as f64 / self.total as f64).min(1.0) as f32 }
    }

    /// "warm-up --blocking 1234/175078 keys (0.7%) | 3.2 keys/s | 15 h 5 min left | 5.1 GB free | 2 giants compiling | 0 failed"
    fn line(&self) -> String {
        let rate = self.rate.map_or_else(|| "measuring rate".into(), |r| format!("{r:.1} {}/s", self.unit));
        let left = if self.stopping {
            "stopping after the current builds".into()
        } else {
            time_left(self.total.saturating_sub(self.done), self.rate)
        };
        format!(
            "{} {}/{} {} ({:.1}%) | {rate} | {left} | {:.1} GB free | {} giant{} compiling | {} failed",
            self.step,
            self.done,
            self.total,
            self.unit,
            self.fraction() * 100.0,
            self.free_mib as f64 / 1024.0,
            self.giants,
            if self.giants == 1 { "" } else { "s" },
            self.failed
        )
    }
}

fn time_left(left: usize, rate: Option<f64>) -> String {
    if left == 0 {
        return "done".into();
    }
    let Some(rate) = rate.filter(|r| *r > 0.0) else { return "time left unknown".into() };
    let s = (left as f64 / rate).ceil() as u64;
    match s {
        0..60 => format!("{s} s left"),
        60..3600 => format!("{} min left", s.div_ceil(60)),
        _ => format!("{} h {} min left", s / 3600, (s % 3600) / 60),
    }
}

unsafe fn run_window(s: &Shared) {
    let Ok(module) = GetModuleHandleW(None) else { return };
    let instance = HINSTANCE(module.0);
    let class = w!("XCLProgressWindow");
    let wc = WNDCLASSW {
        lpfnWndProc: Some(window_proc),
        hInstance: instance,
        lpszClassName: class,
        hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
        hbrBackground: GetSysColorBrush(COLOR_BTNFACE),
        ..Default::default()
    };
    RegisterClassW(&wc);
    let title: Vec<u16> = format!("xcl {}", s.step).encode_utf16().chain([0]).collect();
    let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    let mut frame = RECT { left: 0, top: 0, right: 560, bottom: 112 };
    let _ = AdjustWindowRect(&mut frame, style, false);
    let Ok(hwnd) = CreateWindowExW(WINDOW_EX_STYLE(0), class, PCWSTR(title.as_ptr()), style, CW_USEDEFAULT, CW_USEDEFAULT, frame.right - frame.left, frame.bottom - frame.top, None, None, Some(instance), None) else { return };
    if s.cancel.is_some() {
        if let Ok(button) = CreateWindowExW(WINDOW_EX_STYLE(0), w!("BUTTON"), w!("Cancel"), WS_CHILD | WS_VISIBLE | WS_TABSTOP, 452, 76, 96, 26, Some(hwnd), Some(HMENU(CANCEL_ID as *mut _)), Some(instance), None) {
            SendMessageW(button, WM_SETFONT, Some(WPARAM(GetStockObject(DEFAULT_GUI_FONT).0 as usize)), Some(LPARAM(1)));
        }
    }
    {
        let mut window = s.window.lock().unwrap_or_else(|e| e.into_inner());
        if window.1 {
            drop(window);
            let _ = DestroyWindow(hwnd);
        } else {
            window.0 = hwnd.0 as isize;
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            SetTimer(Some(hwnd), 1, TICK.as_millis() as u32, None);
        }
    }
    let mut msg = MSG::default();
    while GetMessageW(&mut msg, None, 0, 0).as_bool() {
        let _ = TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    s.window.lock().unwrap_or_else(|e| e.into_inner()).0 = 0;
}

fn current() -> Option<Arc<Shared>> {
    CURRENT.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

unsafe fn cancel(hwnd: HWND) {
    let Some(s) = current() else { return };
    let Some(cancel) = s.cancel else { return };
    if !s.cancelled.swap(true, Ordering::Relaxed) {
        cancel();
    }
    if let Ok(button) = GetDlgItem(Some(hwnd), CANCEL_ID as i32) {
        let _ = SetWindowTextW(button, w!("Stopping…"));
    }
}

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_TIMER => {
            let _ = InvalidateRect(Some(hwnd), None, true);
            LRESULT(0)
        }
        WM_COMMAND if wp.0 & 0xffff == CANCEL_ID => {
            cancel(hwnd);
            LRESULT(0)
        }
        // the window closes when the step ends; closing it only minimises it (Cancel stops the step)
        WM_CLOSE => {
            let _ = ShowWindow(hwnd, SW_MINIMIZE);
            LRESULT(0)
        }
        WM_APP_CLOSE => {
            let _ = DestroyWindow(hwnd);
            LRESULT(0)
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            LRESULT(0)
        }
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let dc = BeginPaint(hwnd, &mut ps);
            if !dc.0.is_null() {
                if let Some(s) = current() {
                    let (line, fraction) = s.shown.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    paint(dc, &line, fraction);
                }
            }
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wp, lp),
    }
}

/// The status line split at its first " | " (what and how far, then the rest) and the bar.
unsafe fn paint(dc: HDC, line: &str, fraction: f32) {
    let font = SelectObject(dc, GetStockObject(DEFAULT_GUI_FONT));
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, COLORREF(0x0020_2020));
    let (head, rest) = line.split_once(" | ").unwrap_or((line, ""));
    for (top, text) in [(10, head), (54, rest)] {
        let mut text: Vec<u16> = text.encode_utf16().collect();
        let mut rect = RECT { left: 12, top, right: 548, bottom: top + 18 };
        DrawTextW(dc, &mut text, &mut rect, DT_SINGLELINE | DT_NOPREFIX | DT_END_ELLIPSIS);
    }
    let track = RECT { left: 12, top: 32, right: 548, bottom: 48 };
    let (frame, fill, empty) = (CreateSolidBrush(COLORREF(0x0070_7070)), CreateSolidBrush(COLORREF(0x002e_b82e)), CreateSolidBrush(COLORREF(0x00ff_ffff)));
    FillRect(dc, &track, frame);
    let inner = RECT { left: track.left + 1, top: track.top + 1, right: track.right - 1, bottom: track.bottom - 1 };
    FillRect(dc, &inner, empty);
    let split = inner.left + ((inner.right - inner.left) as f32 * fraction.clamp(0.0, 1.0)).round() as i32;
    FillRect(dc, &RECT { right: split, ..inner }, fill);
    for b in [frame, fill, empty] {
        let _ = DeleteObject(b.into());
    }
    SelectObject(dc, font);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(done: usize, total: usize, rate: Option<f64>) -> Progress {
        Progress { step: "warm-up --blocking", unit: "keys", done, total, failed: 3, rate, free_mib: 5222, giants: 2, stopping: false }
    }

    #[test]
    fn the_line_says_how_far_how_fast_and_what_is_left() {
        assert_eq!(
            progress(1234, 175_078, Some(3.2)).line(),
            "warm-up --blocking 1234/175078 keys (0.7%) | 3.2 keys/s | 15 h 5 min left | 5.1 GB free | 2 giants compiling | 3 failed"
        );
        let first = Progress { giants: 1, ..progress(0, 10, None) };
        assert_eq!(first.line(), "warm-up --blocking 0/10 keys (0.0%) | measuring rate | time left unknown | 5.1 GB free | 1 giant compiling | 3 failed");
        let stopping = Progress { stopping: true, ..progress(5, 10, Some(1.0)) };
        assert!(stopping.line().contains("| stopping after the current builds |"));
        assert_eq!(progress(10, 10, Some(1.0)).fraction(), 1.0);
        assert_eq!(progress(0, 0, None).fraction(), 0.0);
    }

    #[test]
    fn time_left_rounds_up_and_needs_a_rate() {
        assert_eq!(time_left(0, None), "done");
        assert_eq!(time_left(5, None), "time left unknown");
        assert_eq!(time_left(5, Some(0.0)), "time left unknown");
        assert_eq!(time_left(59, Some(1.0)), "59 s left");
        assert_eq!(time_left(61, Some(1.0)), "2 min left");
        assert_eq!(time_left(7260, Some(1.0)), "2 h 1 min left");
    }

    #[test]
    fn the_rate_is_over_the_recent_samples_once_they_span_5_s() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut samples: VecDeque<_> = [(at(0), 0), (at(4), 40)].into();
        assert_eq!(rate(&samples), None);
        samples.push_back((at(10), 100));
        assert_eq!(rate(&samples), Some(10.0));
        assert_eq!(rate(&VecDeque::new()), None);
    }
}
