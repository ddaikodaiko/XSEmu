//! Write32/64BitValueBottomOfPipeX, FillMemoryWith*X and CopyMemoryToMemoryX into memory the GPU
//! does not see: done by the CPU once the GPU has finished the work before them.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! Each queue signals its own fence and publishes writes in submission order once that fence
//! completes. An incomplete submission on one queue doesn't block another queue's writes.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use windows::core::{Interface, Owned};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::CreateEventW;

use crate::gpu::fence::{self, FenceError};
use crate::memmap::Access;

/// What the CPU does for the GPU once the work before it is done.
#[derive(Clone)]
pub(super) enum Write {
    /// the dwords from an address
    Immediate(usize, Vec<u32>),
    /// `bytes` from an address repeating a 16-byte pattern
    Fill(usize, [u8; 16], usize),
    /// `bytes` from `src` to `dst`
    Copy { dst: usize, src: usize, bytes: usize },
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
    DispatchTiming {
        shader: u64,
        at: Option<(usize, u32, usize)>,
        groups: [u32; 3],
        src: usize,
        _buf: ID3D12Resource,
    },
}

struct Job {
    value: u64,
    writes: Vec<Write>,
    timestamp_hz: u64,
    calibration: Option<(u64, u64, i64)>,
}

struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

const WAIT_MS: u32 = 10_000;

fn spawn_queue_worker(dev: ID3D12Device, fence: ID3D12Fence) -> std::io::Result<mpsc::Sender<Job>> {
    let (tx, rx) = mpsc::channel::<Job>();
    std::thread::Builder::new().name("xcl-cpu-writes".into()).spawn(move || unsafe {
        let event = CreateEventW(None, false, false, None).ok().map(|event| Owned::new(event));
        for job in rx {
            let complete = match event.as_deref().copied().map(|event| fence::wait(&fence, job.value, &dev, Some(event), WAIT_MS)) {
                Some(Ok(())) => true,
                Some(Err(FenceError::Removed(_))) => return,
                _ => false,
            };
            if !complete {
                let mut first_pending = true;
                loop {
                    match fence::completed_value(&fence, &dev) {
                        Ok(value) if value >= job.value => break,
                        Ok(value) => {
                            if std::mem::take(&mut first_pending) {
                                static DEFERRED: AtomicU64 = AtomicU64::new(0);
                                let total = DEFERRED.fetch_add(1, Ordering::Relaxed) + 1;
                                if total.is_power_of_two() {
                                    crate::xlog_warn!("D3D12.X: deferred GPU writes still waiting ({total} jobs so far; fence at {value}, needed {}; {} writes held)", job.value, job.writes.len());
                                }
                            }
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        Err(_) => return,
                    }
                }
            }
            for w in job.writes {
                write_now(w, job.timestamp_hz, job.calibration);
            }
        }
    })?;
    Ok(tx)
}

pub(super) fn forget_queue(queue: usize) {
    if let Some(m) = FENCES.lock().unwrap().as_mut() {
        m.remove(&queue);
    }
}

/// What `[va, va + bytes)` allows the CPU now.
fn access(va: usize, bytes: usize) -> Access {
    if let Some(a) = crate::memmap::access(va, bytes) {
        return a;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0;
    if !found || m.State != MEM_COMMIT || m.BaseAddress as usize + m.RegionSize < va + bytes {
        return Access::None;
    }
    Access::of(m.Protect)
}

/// `w`, if its memory is still there.
unsafe fn write_now(w: Write, timestamp_hz: u64, calibration: Option<(u64, u64, i64)>) {
    match w {
        Write::Immediate(va, words) => {
            if access(va, 4 * words.len()) != Access::Write {
                return;
            }
            for (i, &w) in words.iter().enumerate() {
                (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
            }
            super::wait_trace::written(va, &words);
        }
        Write::Fill(va, pattern, bytes) => {
            if access(va, bytes) != Access::Write {
                return;
            }
            for (i, b) in std::slice::from_raw_parts_mut(va as *mut u8, bytes).iter_mut().enumerate() {
                *b = pattern[i % 16];
            }
        }
        Write::Copy { dst, src, bytes } => {
            if access(dst, bytes) != Access::Write || access(src, bytes) == Access::None {
                return;
            }
            std::ptr::copy(src as *const u8, dst as *mut u8, bytes);
        }
        Write::ReadbackU64 { va, src, .. } => {
            if access(va, 8) != Access::Write {
                return;
            }
            let value = (*(src as *const AtomicU64)).load(Ordering::Acquire);
            if va % 8 == 0 {
                (*(va as *const AtomicU64)).store(value, Ordering::Release);
            } else {
                (va as *mut u64).write_unaligned(value);
            }
        }
        Write::DispatchTiming { shader, at, groups: [x, y, z], src, .. } => {
            if timestamp_hz == 0 {
                return;
            }
            let ticks = src as *const u64;
            let start = ticks.read_volatile();
            let end = ticks.add(1).read_volatile();
            if start == 0 || end < start {
                return;
            }
            let ms = (end - start) as f64 * 1000.0 / timestamp_hz as f64;
            let at = at.map_or(String::new(), |(slot, generation, op)| format!(" slot {slot}/{generation} op {}", op + 1));
            crate::xlog_info!("GPU dispatch timing: CS {shader:016x}{at} groups {x}x{y}x{z}: {ms:.3} ms");
            if let Some((gpu, qpc, qpc_hz)) = calibration {
                crate::xlog_info!("GPU dispatch clock: CS {shader:016x}{at} groups {x}x{y}x{z}: gpu_start={start} gpu_end={end} gpu_ref={gpu} qpc_ref={qpc} gpu_hz={timestamp_hz} qpc_hz={qpc_hz}");
            }
        }
    }
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    crate::hle::kernelx::lazy::ensure(va, bytes);
    access(va, bytes) == Access::Write
}

/// Whether `[va, va + bytes)` is memory the CPU can read for the GPU later.
pub(super) fn readable(va: usize, bytes: usize) -> bool {
    crate::hle::kernelx::lazy::ensure(va, bytes);
    access(va, bytes) != Access::None
}

/// `writes` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, writes: Vec<Write>) {
    if writes.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let timestamp_hz = if writes.iter().any(|w| matches!(w, Write::DispatchTiming { .. })) {
        q.GetTimestampFrequency().unwrap_or(0)
    } else {
        0
    };
    static QPC_TRACE: OnceLock<bool> = OnceLock::new();
    let calibration = if timestamp_hz != 0 && *QPC_TRACE.get_or_init(|| {
        std::env::var("XCL_GPU_DISPATCH_QPC").is_ok_and(|s| s.trim() == "1")
    }) {
        let (mut gpu, mut qpc, mut qpc_hz) = (0, 0, 0);
        (windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut qpc_hz) != 0
            && qpc_hz > 0 && q.GetClockCalibration(&mut gpu, &mut qpc).is_ok())
            .then_some((gpu, qpc, qpc_hz))
    } else {
        None
    };
    let mut g = FENCES.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    let entry = match m.entry(queue as usize) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => {
            let mut dev: Option<ID3D12Device> = None;
            if q.GetDevice(&mut dev).is_err() {
                return;
            }
            let Some(dev) = dev else { return };
            let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
            let tx = match spawn_queue_worker(dev, fence.clone()) {
                Ok(tx) => tx,
                Err(e) => {
                    crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
                    return;
                }
            };
            v.insert(QueueFence { fence, value: 0, tx })
        }
    };
    entry.value += 1;
    if q.Signal(&entry.fence, entry.value).is_ok() {
        let _ = entry.tx.send(Job { value: entry.value, writes, timestamp_hz, calibration });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::Immediate(va, vec![7, 9])]);
            let t = std::time::Instant::now();
            while (*((va + 4) as *const AtomicU32)).load(Ordering::Acquire) != 9 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// Fills and copies land in order after the queue: the copy sees the fill before it.
    #[test]
    fn fills_and_copies_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let src = Box::new([0u8; 32]);
            let dst = Box::new([0u32; 9]);
            let (s, d) = (src.as_ptr() as usize, dst.as_ptr() as usize);
            assert!(readable(s, 32) && writable(d, 36));
            let pattern: [u8; 16] = std::array::from_fn(|i| i as u8);
            after(q.as_raw(), vec![Write::Fill(s, pattern, 32), Write::Copy { dst: d, src: s, bytes: 32 }, Write::Immediate(d + 32, vec![1])]);
            let t = std::time::Instant::now();
            while (*((d + 32) as *const AtomicU32)).load(Ordering::Acquire) != 1 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            let copied = std::slice::from_raw_parts(d as *const u8, 32);
            assert!(copied.iter().enumerate().all(|(i, &b)| b == (i % 16) as u8));
        }
    }

    /// A ReadbackU64 write copies the 64-bit value from the mapped readback buffer after the queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Write::ReadbackU64 { va, src: ptr as usize, _buf: rb }]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }
}
