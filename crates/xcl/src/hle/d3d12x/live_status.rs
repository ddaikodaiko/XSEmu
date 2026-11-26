use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
use windows_sys::Win32::System::Performance::*;
use windows_sys::Win32::System::Threading::*;

use crate::live_stall::{self, Sample, Stage};

#[derive(Default)]
struct Metrics {
    at: Option<Instant>,
    cpu: Option<f64>,
    cores: Vec<(String, f64)>,
    engines: Vec<(String, f64)>,
    vram: Option<(u64, u64)>,
}

static METRICS: Mutex<Metrics> = Mutex::new(Metrics {
    at: None,
    cpu: None,
    cores: Vec::new(),
    engines: Vec::new(),
    vram: None,
});
static RUN: AtomicBool = AtomicBool::new(false);

pub(super) fn advanced() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_LIVE_TITLE").as_deref() != Ok("0"))
}

struct Pdh {
    query: PDH_HQUERY,
    cpu: PDH_HCOUNTER,
    gpu: PDH_HCOUNTER,
    buffer: Vec<PDH_FMT_COUNTERVALUE_ITEM_W>,
}

impl Pdh {
    fn new(details: bool) -> Self {
        let mut query = std::ptr::null_mut();
        let mut cpu = std::ptr::null_mut();
        let mut gpu = std::ptr::null_mut();
        unsafe {
            if PdhOpenQueryW(std::ptr::null(), 0, &mut query) == 0 {
                if details {
                    PdhAddEnglishCounterW(
                        query,
                        w!(r"\Processor Information(*)\% Processor Time").0,
                        0,
                        &mut cpu,
                    );
                }
                PdhAddEnglishCounterW(
                    query,
                    w!(r"\GPU Engine(*)\Utilization Percentage").0,
                    0,
                    &mut gpu,
                );
            }
        }
        Self {
            query,
            cpu,
            gpu,
            buffer: Vec::new(),
        }
    }

    fn read(&mut self, counter: PDH_HCOUNTER, mut visit: impl FnMut(String, f64)) {
        if counter.is_null() {
            return;
        }
        unsafe {
            let mut bytes = (self.buffer.len() * size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>()) as u32;
            let mut count = 0;
            let ptr = if bytes == 0 {
                std::ptr::null_mut()
            } else {
                self.buffer.as_mut_ptr()
            };
            let mut status =
                PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, ptr);
            if status == PDH_MORE_DATA && bytes > 0 && bytes <= 2 * 1024 * 1024 {
                self.buffer.resize(
                    (bytes as usize).div_ceil(size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>()),
                    Default::default(),
                );
                status = PdhGetFormattedCounterArrayW(
                    counter,
                    PDH_FMT_DOUBLE,
                    &mut bytes,
                    &mut count,
                    self.buffer.as_mut_ptr(),
                );
            }
            if status != 0 || count as usize > self.buffer.len() {
                return;
            }
            let end = self.buffer.as_ptr_range().end as usize;
            for item in &self.buffer[..count as usize] {
                if !matches!(
                    item.FmtValue.CStatus,
                    PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA
                ) {
                    continue;
                }
                let p = item.szName as usize;
                if p < self.buffer.as_ptr() as usize || p >= end || p % align_of::<u16>() != 0 {
                    continue;
                }
                let name = std::slice::from_raw_parts(item.szName, ((end - p) / 2).min(256));
                let Some(len) = name.iter().position(|&c| c == 0) else {
                    continue;
                };
                let value = item.FmtValue.Anonymous.doubleValue;
                if value.is_finite() && value >= 0.0 {
                    visit(String::from_utf16_lossy(&name[..len]), value.min(100.0));
                }
            }
        }
    }
}

impl Drop for Pdh {
    fn drop(&mut self) {
        if !self.query.is_null() {
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}

fn process_ticks() -> Option<u64> {
    let [mut create, mut exit, mut kernel, mut user] = [FILETIME::default(); 4];
    let ticks = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    (unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut create,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    } != 0)
        .then(|| ticks(kernel) + ticks(user))
}

pub(super) fn start() {
    RUN.store(true, Ordering::Release);
    let _ = std::thread::Builder::new()
        .name("xcl-live-status".into())
        .spawn(sample);
}

pub(super) fn stop() {
    RUN.store(false, Ordering::Release);
}

fn sample() {
    let details = live_stall::overlay();
    let metrics = details || advanced();
    let mut pdh = metrics.then(|| Pdh::new(details));
    let logical = unsafe { GetActiveProcessorCount(u16::MAX) }.max(1) as f64;
    let pid = unsafe { GetCurrentProcessId() };
    let mut previous = None;
    let mut last_log = None;
    while RUN.load(Ordering::Acquire) {
        let at = Instant::now();
        let ticks = metrics.then(process_ticks).flatten();
        let cpu = ticks
            .zip(previous)
            .and_then(|(ticks, (t, old)): (u64, (Instant, u64))| {
                let seconds = at.duration_since(t).as_secs_f64();
                (seconds > 0.0).then(|| {
                    (ticks.saturating_sub(old) as f64 / (seconds * 1e5 * logical)).min(100.0)
                })
            });
        previous = ticks.map(|t| (at, t));
        let mut cores = BTreeMap::new();
        let mut engines = BTreeMap::new();
        if let Some(pdh) = pdh
            .as_mut()
            .filter(|p| !p.query.is_null() && unsafe { PdhCollectQueryData(p.query) } == 0)
        {
            pdh.read(pdh.cpu, |name, value| {
                if let Some((group, core)) = name
                    .split_once(',')
                    .and_then(|(g, c)| Some((g.parse::<u16>().ok()?, c.parse::<u16>().ok()?)))
                {
                    cores.insert((group, core), value);
                }
            });
            if let Some(luid) = super::real::video_adapter_luid() {
                let prefix = format!(
                    "pid_{pid}_luid_0x{:08x}_0x{:08x}_",
                    luid.HighPart as u32, luid.LowPart
                );
                pdh.read(pdh.gpu, |name, value| {
                    let name = name.to_ascii_lowercase();
                    let Some((identity, kind)) = name
                        .strip_prefix(&prefix)
                        .and_then(|s| s.split_once("_engtype_"))
                    else {
                        return;
                    };
                    let Some((node, engine)) = identity
                        .strip_prefix("phys_")
                        .and_then(|s| s.split_once("_eng_"))
                    else {
                        return;
                    };
                    let (Ok(node), Ok(engine)) = (node.parse::<u16>(), engine.parse::<u16>())
                    else {
                        return;
                    };
                    if !kind.is_empty()
                        && kind.len() <= 32
                        && kind.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                    {
                        let total = engines
                            .entry(format!("{kind} {node}:{engine}"))
                            .or_insert(0.0_f64);
                        *total = (*total + value).min(100.0);
                    }
                });
            }
        }
        let vram = metrics
            .then(super::real::video_memory_info)
            .flatten()
            .map(|[local, _]| {
                (
                    local.CurrentUsage,
                    local.Budget.saturating_sub(local.CurrentUsage),
                )
            });
        let mut engines: Vec<_> = engines.into_iter().collect();
        engines.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let cores = cores
            .into_iter()
            .map(|((g, c), value)| (format!("{g}:{c}"), value))
            .collect();
        if let Ok(mut m) = METRICS.lock() {
            *m = Metrics {
                at: Some(at),
                cpu,
                cores,
                engines,
                vram,
            };
        }
        if live_stall::logs()
            && last_log
                .is_none_or(|last: Instant| at.duration_since(last) >= Duration::from_secs(5))
        {
            let spans = observations();
            let active: Vec<_> = spans
                .iter()
                .filter(|s| at.saturating_duration_since(s.since) >= Duration::from_secs(1))
                .filter(|s| {
                    s.stage.blocking()
                        || matches!(
                            s.stage,
                            Stage::DriverSlot
                                | Stage::DriverBuild
                                | Stage::ShaderCompile
                                | Stage::Queued
                        )
                })
                .take(4)
                .map(|s| s.text(at))
                .collect();
            if !active.is_empty() {
                crate::xlog_info!("live stall: {}", active.join("; "));
            }
            last_log = Some(at);
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn observations() -> Vec<Sample> {
    let mut spans = live_stall::snapshot();
    spans.extend(super::pso_async::diagnostic_queued());
    spans.extend(super::list_x::diagnostic_gpu_wait());
    spans.sort_unstable_by_key(|s| s.since);
    spans
}

fn format_metrics(m: &Metrics) -> String {
    let cpu = m
        .cpu
        .map_or_else(|| "unavailable".into(), |v| format!("{v:.0}%"));
    let gpu = m
        .engines
        .first()
        .map_or_else(|| "unavailable".into(), |(_, v)| format!("{v:.0}%"));
    let vram = m.vram.map_or_else(
        || "unavailable".into(),
        |(used, head)| {
            format!(
                "{} MiB used, {} MiB budget headroom",
                used >> 20,
                head >> 20
            )
        },
    );
    format!("XCL CPU {cpu} | XCL GPU (peak engine) {gpu} | Local VRAM {vram}")
}

fn fresh(m: &Metrics) -> bool {
    m.at.is_some_and(|at| at.elapsed() < Duration::from_secs(4))
}

pub(super) fn title() -> String {
    METRICS.try_lock().ok().filter(|m| fresh(m)).map_or_else(
        || format_metrics(&Metrics::default()),
        |m| format_metrics(&m),
    )
}

fn pinned_core(tid: u32) -> Option<u32> {
    if tid == 0 || unsafe { GetActiveProcessorGroupCount() } != 1 {
        return None;
    }
    let thread = unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) };
    if thread.is_null() {
        return None;
    }
    let mut affinity = Default::default();
    let ok = unsafe { GetThreadGroupAffinity(thread, &mut affinity) } != 0;
    unsafe { CloseHandle(thread) };
    (ok && affinity.Mask.count_ones() == 1).then(|| affinity.Mask.trailing_zeros())
}

fn panel_text(fps: f64) -> Vec<String> {
    let now = Instant::now();
    let mut lines = vec![format!(
        "XCL LIVE   |   {fps:.1} host FPS   |   F10 hide/show"
    )];
    let mut detail = Vec::new();
    if let Some(m) = METRICS.try_lock().ok().filter(|m| fresh(m)) {
        lines.push(format_metrics(&m));
        if m.engines.is_empty() {
            lines.push("GPU engines unavailable for this process / adapter".into());
        } else {
            lines.push(format!(
                "Busiest XCL GPU engine: {} ({:.0}%)",
                m.engines[0].0, m.engines[0].1
            ));
            for group in m.engines.chunks(4) {
                detail.push(
                    group
                        .iter()
                        .map(|(k, v)| format!("{k} {v:.0}%"))
                        .collect::<Vec<_>>()
                        .join("   "),
                );
            }
        }
        if m.cores.is_empty() {
            lines.push("Per-logical CPU (system): unavailable".into());
        } else {
            let mut busiest: Vec<_> = m.cores.iter().collect();
            busiest.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            lines.push(format!(
                "Busiest logical CPUs (system, group:CPU): {}",
                busiest
                    .into_iter()
                    .take(3)
                    .map(|(k, v)| format!("{k} {v:.0}%"))
                    .collect::<Vec<_>>()
                    .join("   ")
            ));
            for group in m.cores.chunks(8) {
                detail.push(
                    group
                        .iter()
                        .map(|(k, v)| format!("{k} {v:.0}%"))
                        .collect::<Vec<_>>()
                        .join("   "),
                );
            }
        }
    } else {
        lines.push("OS counters unavailable / awaiting sample".into());
    }
    let spans = observations();
    if let Some(wait) = spans.iter().find(|s| {
        s.stage.blocking() && now.saturating_duration_since(s.since) >= Duration::from_millis(250)
    }) {
        lines.push(format!("Observed wait: {}", wait.text(now)));
        if let Some(core) = pinned_core(wait.tid) {
            lines.push(format!(
                "TID {} single-CPU affinity observed: 0:{core}",
                wait.tid
            ));
        }
        if wait.pipeline != 0 {
            if let Some(stage) = spans
                .iter()
                .filter(|s| s.pipeline == wait.pipeline && !s.stage.blocking())
                .max_by_key(|s| (s.stage != Stage::PipelineBuild, s.since))
            {
                lines.push(format!("Waited PSO: {}", stage.text(now)));
            }
        }
    } else {
        lines.push("Blocking reason: unknown (no long tracked wait observed)".into());
    }
    let p = crate::progress::snapshot();
    lines.push(format!(
        "Pipeline work {} | translation {} | DXC {} | driver slots {}",
        p.pipelines.pending, p.translations.pending, p.dxc.pending, p.driver.pending
    ));
    for s in spans
        .iter()
        .filter(|s| !s.stage.blocking() && s.stage != Stage::PipelineBuild)
        .take(4)
    {
        lines.push(s.text(now));
    }
    for s in spans
        .iter()
        .filter(|s| s.stage == Stage::PipelineBuild)
        .take(2)
    {
        lines.push(s.text(now));
    }
    lines.push("Per-CUDA-core load / stuck hardware core: unavailable".into());
    lines.extend(detail);
    lines
}

struct Panel {
    hwnd: HWND,
    shown: bool,
    lines: Vec<String>,
}

thread_local! {
    static PANEL: RefCell<Option<Panel>> = const { RefCell::new(None) };
}

pub(super) unsafe fn install(owner: HWND, instance: HINSTANCE) {
    if !live_stall::overlay() {
        return;
    }
    let name = w!("XCLLiveStallPanel");
    let class = WNDCLASSW {
        lpfnWndProc: Some(paint),
        hInstance: instance,
        lpszClassName: name,
        ..Default::default()
    };
    if RegisterClassW(&class) == 0 {
        return;
    }
    if let Ok(hwnd) = CreateWindowExW(
        WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT | WS_EX_LAYERED,
        name,
        PCWSTR::null(),
        WS_POPUP,
        0,
        0,
        1,
        1,
        Some(owner),
        None,
        Some(instance),
        None,
    ) {
        if SetLayeredWindowAttributes(hwnd, COLORREF(0), 224, LWA_ALPHA).is_ok() {
            PANEL.with(|p| {
                *p.borrow_mut() = Some(Panel {
                    hwnd,
                    shown: true,
                    lines: Vec::new(),
                })
            });
        } else {
            let _ = DestroyWindow(hwnd);
        }
    }
}

pub(super) unsafe fn key(msg: u32, lp: LPARAM) -> bool {
    let Some((hwnd, shown)) = PANEL.with(|p| {
        let mut p = p.borrow_mut();
        let panel = p.as_mut()?;
        if matches!(msg, WM_KEYDOWN | WM_SYSKEYDOWN) && lp.0 & (1 << 30) == 0 {
            panel.shown = !panel.shown;
        }
        Some((panel.hwnd, panel.shown))
    }) else {
        return false;
    };
    if !shown {
        let _ = ShowWindow(hwnd, SW_HIDE);
    }
    true
}

pub(super) unsafe fn refresh(owner: HWND, fps: f64) {
    let Some((hwnd, shown)) = PANEL.with(|p| p.borrow().as_ref().map(|p| (p.hwnd, p.shown))) else {
        return;
    };
    if !shown
        || IsIconic(owner).as_bool()
        || GetForegroundWindow() != owner
        || !IsWindowVisible(owner).as_bool()
    {
        if IsWindowVisible(hwnd).as_bool() {
            let _ = ShowWindow(hwnd, SW_HIDE);
        }
        return;
    }
    let mut rc = RECT::default();
    let mut pt = POINT { x: 8, y: 8 };
    if GetClientRect(owner, &mut rc).is_err() || !ClientToScreen(owner, &mut pt).as_bool() {
        return;
    }
    let lines = panel_text(fps);
    let h = (20 * lines.len() as i32 + 14).min(rc.bottom - 16);
    let width = (rc.right - 16).min(900);
    if h <= 0 || width <= 0 {
        let _ = ShowWindow(hwnd, SW_HIDE);
        return;
    }
    PANEL.with(|p| {
        if let Some(panel) = p.borrow_mut().as_mut() {
            panel.lines = lines;
        }
    });
    let _ = SetWindowPos(
        hwnd,
        None,
        pt.x,
        pt.y,
        width,
        h,
        SWP_NOACTIVATE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_SHOWWINDOW,
    );
    let _ = InvalidateRect(Some(hwnd), None, false);
}

unsafe extern "system" fn paint(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let dc = BeginPaint(hwnd, &mut ps);
            let mut area = RECT::default();
            if !dc.0.is_null() && GetClientRect(hwnd, &mut area).is_ok() {
                FillRect(dc, &area, HBRUSH(GetStockObject(BLACK_BRUSH).0));
                let font = SelectObject(dc, GetStockObject(DEFAULT_GUI_FONT));
                SetBkMode(dc, TRANSPARENT);
                SetTextColor(dc, COLORREF(0x00f5f0e4));
                PANEL.with(|p| {
                    if let Some(panel) = p.borrow().as_ref() {
                        for (i, line) in panel.lines.iter().enumerate() {
                            let top = 7 + 20 * i as i32;
                            if top + 20 > area.bottom {
                                break;
                            }
                            let mut rect = RECT {
                                left: 9,
                                top,
                                right: area.right - 9,
                                bottom: top + 20,
                            };
                            let mut text: Vec<_> = line.encode_utf16().collect();
                            DrawTextW(
                                dc,
                                &mut text,
                                &mut rect,
                                DT_SINGLELINE | DT_NOPREFIX | DT_END_ELLIPSIS,
                            );
                        }
                    }
                });
                SelectObject(dc, font);
            }
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        }
        _ => DefWindowProcW(hwnd, msg, wp, lp),
    }
}
