//! Memory committed on first use: physical pages mapped as copies but not used yet, and the views
//! of physical pages' sections (`kernelx::physical`).
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a copy of a page whose contents are still zero is mapped as reserved
//! address space only: the CPU's first access faults, the page is committed – with what the
//! physical page holds by then (its section, written through a view meanwhile) – and the access
//! goes on. A section's pages are reserved too (SEC_RESERVE): the first access through any view
//! commits the page for all of them. Kernel calls do not fault in the title – they fail – so the
//! buffers of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{
    GetLastError, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY, ERROR_NO_SYSTEM_RESOURCES, ERROR_WORKING_SET_QUOTA, EXCEPTION_ACCESS_VIOLATION, NTSTATUS,
};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet → where their contents are (0: zeros)
static LAZY: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::new());
/// views of physical pages' sections: start → end
static VIEWS: RwLock<BTreeMap<usize, usize>> = RwLock::new(BTreeMap::new());

/// The handler is in place (d3d12x::image_writeback guards pages it answers for).
pub(crate) fn install_handler() {
    install();
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

fn committed(a: usize) -> bool {
    if let Some(x) = crate::memmap::access(a, 1) {
        return x != crate::memmap::Access::None;
    }
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    unsafe { VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT }
}

/// Whether a commit Windows refused with `error` can succeed a moment later: the system's commit or
/// the working set was short, which passes when another allocation is freed. (An address that
/// cannot be committed at all stays refused.)
fn refusal_passes(error: u32) -> bool {
    matches!(error, ERROR_NOT_ENOUGH_MEMORY | ERROR_COMMITMENT_LIMIT | ERROR_WORKING_SET_QUOTA | ERROR_NO_SYSTEM_RESOURCES)
}

/// `len` bytes at `a` committed read-write (false: Windows refused). The console's pages are
/// backed from their allocation on, so an access to one never fails; here the backing is committed
/// at first use, and a refusal that passes (`refusal_passes`) is asked again for about half a
/// second before the access or the call it is for fails.
pub(super) unsafe fn commit_memory(a: usize, len: usize) -> bool {
    static REFUSED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let mut wait = 1;
    loop {
        if !VirtualAlloc(a as *const c_void, len, MEM_COMMIT, PAGE_READWRITE).is_null() {
            if wait > 1 && REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                crate::xlog_info!("memory: Windows committed {len:#x} bytes at {a:#x} after refusing for up to {} ms (commit short)", wait - 1);
            }
            return true;
        }
        let error = GetLastError();
        if !refusal_passes(error) || wait > 256 {
            if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                crate::xlog_warn!("memory: Windows refused to commit {len:#x} bytes at {a:#x} (error {error}): the access or call it is for fails");
            }
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(wait));
        wait *= 2;
    }
}

/// Commit `page` if it is a lazy one, with its contents (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let Some(&from) = g.get(&page) else { return false };
    if !commit_memory(page, PAGE) {
        return false;
    }
    crate::memmap::committed(page, PAGE, PAGE_READWRITE);
    if crate::gpu::uma::get().is_some_and(|u| u.is_sparse(page)) {
        crate::hle::d3d12x::sparse::committed(page, PAGE);
    }
    if from != 0 && super::physical::used(from) {
        std::ptr::copy_nonoverlapping(from as *const u8, page as *mut u8, PAGE);
        crate::eh::note_guest_write("physical page copy filled on first use", page, PAGE);
    }
    g.remove(&page);
    true
}

/// What the memory at `a` is, for a crash report.
pub fn describe(a: usize) -> String {
    let page = a & !(PAGE - 1);
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let os = if unsafe { VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 {
        format!("Windows: state {:#x} protect {:#x} type {:#x}, region {:#x}+{:#x} of allocation {:#x}", m.State, m.Protect, m.Type, m.BaseAddress as usize, m.RegionSize, m.AllocationBase as usize)
    } else {
        "Windows: no answer".into()
    };
    let lazy = LAZY.read().map(|g| g.contains_key(&page)).unwrap_or(false);
    format!("{os}; XCL's record {:?}, lazy {lazy}, section view {}", crate::memmap::access(a, 1), in_view(page))
}

fn in_view(a: usize) -> bool {
    VIEWS.read().unwrap_or_else(|e| e.into_inner()).range(..=a).next_back().is_some_and(|(_, &end)| a < end)
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if crate::hle::d3d12x::image_writeback::fault(r.ExceptionInformation[1]) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    let page = r.ExceptionInformation[1] & !(PAGE - 1);
    let lazy = LAZY.read().unwrap_or_else(|e| e.into_inner()).contains_key(&page);
    if lazy && commit(page) || allows(r.ExceptionInformation[0], r.ExceptionInformation[1]) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // a section's page used for the first time
    if in_view(page) && commit_memory(page, PAGE) {
        crate::memmap::committed(page, PAGE, PAGE_READWRITE);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    EXCEPTION_CONTINUE_SEARCH
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 {
        return false;
    }
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let reads = writes | PAGE_READONLY | PAGE_EXECUTE_READ;
    match access {
        0 => m.Protect & reads != 0,
        1 => m.Protect & writes != 0,
        _ => false,
    }
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use, with what `from` holds
/// then (None: zeros).
pub(crate) fn mark(page: usize, from: Option<usize>) {
    install();
    LAZY.write().unwrap_or_else(|e| e.into_inner()).insert(page, from.unwrap_or(0));
}

/// Whether `page` waits for its first use.
pub(crate) fn is_lazy(page: usize) -> bool {
    LAZY.read().unwrap_or_else(|e| e.into_inner()).contains_key(&page)
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    LAZY.write().unwrap_or_else(|e| e.into_inner()).remove(&page).is_some()
}

/// The lazy pages filled from `[start, start + len)` (a section's view going away) are zeros.
pub(super) fn forget_fills(start: usize, len: usize) {
    for from in LAZY.write().unwrap_or_else(|e| e.into_inner()).values_mut() {
        if *from >= start && *from < start + len {
            *from = 0;
        }
    }
}

/// `[start, start + len)` is a view of a section: its pages are committed on first use.
pub(super) fn view(start: usize, len: usize) {
    install();
    VIEWS.write().unwrap_or_else(|e| e.into_inner()).insert(start, start + len);
}

/// The view at `start` is gone.
pub(super) fn unview(start: usize) {
    VIEWS.write().unwrap_or_else(|e| e.into_inner()).remove(&start);
}

/// Whether `[p, p + len)` overlaps a view of a section.
pub(crate) fn overlaps_view(p: usize, len: usize) -> bool {
    let g = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    if g.is_empty() {
        return false;
    }
    let end = p.saturating_add(len);
    g.range(..end).next_back().is_some_and(|(_, &e)| e > p)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes – and the
/// views' pages there.
pub fn ensure(p: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = p.saturating_add(len);
    let pages: Vec<usize> = {
        let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() { Vec::new() } else { g.range(p & !(PAGE - 1)..end).map(|(&k, _)| k).collect() }
    };
    for page in pages {
        unsafe { commit(page) };
    }
    let views: Vec<(usize, usize)> = {
        let g = VIEWS.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        let first = g.range(..=p).next_back().map_or(p, |(&s, _)| s);
        g.range(first..end).filter(|(_, &e)| e > p).map(|(&s, &e)| (s.max(p & !(PAGE - 1)), e.min(end))).collect()
    };
    for (lo, hi) in views {
        let mut page = lo & !(PAGE - 1);
        while page < hi {
            if !committed(page) && unsafe { commit_memory(page, PAGE) } {
                crate::memmap::committed(page, PAGE, PAGE_READWRITE);
            }
            page += PAGE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A commit refused because memory was short is asked again; one refused for its address is
    /// not. Memory that can be committed is, at the first asking.
    #[test]
    fn a_commit_refused_for_want_of_memory_is_asked_again() {
        assert!([ERROR_NOT_ENOUGH_MEMORY, ERROR_COMMITMENT_LIMIT, ERROR_WORKING_SET_QUOTA, ERROR_NO_SYSTEM_RESOURCES].into_iter().all(refusal_passes));
        assert!(!refusal_passes(windows_sys::Win32::Foundation::ERROR_INVALID_ADDRESS) && !refusal_passes(0));
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert!(commit_memory(r, PAGE));
            std::ptr::write_volatile(r as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile(r as *const u32), 7);
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault. (Filling from a used physical page: `physical`'s and `dstorage`'s tests.)
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r, None);
            mark(r + PAGE, None);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!LAZY.read().unwrap().contains_key(&(r + PAGE)));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            mark(r + 2 * PAGE, Some(r));
            assert_eq!(std::ptr::read_volatile((r + 2 * PAGE + 8) as *const u32), 0, "a source that is no used physical page: zeros");
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// A first use in memory XCL did not see reserved (it will not see it released either) leaves
    /// nothing in XCL's record of title memory: a span kept there answered "committed" for
    /// whatever Windows put at the address next (a fresh reservation read as committed, and a
    /// page mapped there was copied into memory that was not).
    #[test]
    fn a_first_use_in_memory_xcl_did_not_reserve_leaves_no_record() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r, None);
            std::ptr::write_volatile(r as *mut u32, 7);
            assert_eq!(crate::memmap::access(r, 4), None, "asked of Windows");
            assert!(committed(r) && !committed(r + PAGE));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
            // (the address reserved again, if no other thread took it meanwhile)
            if VirtualAlloc(r as *const c_void, 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize == r {
                assert!(!committed(r), "reserved, not committed");
                VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
            }
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE, None);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
