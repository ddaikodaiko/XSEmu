//! XCL's own record of the title's memory: what XCL reserved and committed for it – the
//! title's allocations through kernelx, physical pages and their views, unified memory and its
//! sparse windows – so the hot paths that ask whether title memory is there (`gpu::shader::
//! readable` per draw, texture uploads, DirectStorage destinations) need no system call.
//! VirtualQuery is a system call that walks the address space's tree and sizes a region by
//! walking its pages: with hundreds of thousands of views and placeholders mapped, and views of
//! gigabytes, one call took milliseconds. Addresses outside everything recorded (host DLLs, heaps,
//! stacks) are still asked of Windows (`access` returns None).
//!
//! The record must never say more than Windows has: an address answered "committed" that is not
//! is written by XCL (a page mapped as a copy, a DirectStorage destination) and faults. So a
//! commit is kept only inside address space XCL saw reserved – whose release it sees too
//! (`released`) – or in unified memory, which is never released; and a new reservation starts
//! with nothing recorded in it, whatever was there before (address space Windows hands out is
//! free: a reservation recorded there went back without XCL seeing it).

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOPY};

/// What the memory allows, least first.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Access {
    /// not committed, or no access
    None,
    Read,
    Write,
}

impl Access {
    /// From a Windows page protection.
    pub fn of(protect: u32) -> Access {
        if protect == 0 || protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
            Access::None
        } else if protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) != 0 {
            Access::Write
        } else {
            Access::Read
        }
    }
}

#[derive(Default)]
struct Map {
    /// address space XCL reserved or handed out: start → end (allocations, not merged)
    known: BTreeMap<usize, usize>,
    /// committed memory: start → (end, access), merged where the access is the same
    committed: BTreeMap<usize, (usize, Access)>,
}

static MAP: RwLock<Map> = RwLock::new(Map { known: BTreeMap::new(), committed: BTreeMap::new() });

/// calls answered from the record, and asked of Windows
static ANSWERED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);

impl Map {
    fn cut(&mut self, a: usize, b: usize) {
        let hit: Vec<(usize, (usize, Access))> = self.committed.range(..b).rev().take_while(|(_, &(e, _))| e > a).map(|(&s, &v)| (s, v)).collect();
        for (s, (e, acc)) in hit {
            self.committed.remove(&s);
            if s < a {
                self.committed.insert(s, (a, acc));
            }
            if e > b {
                self.committed.insert(b, (e, acc));
            }
        }
    }

    fn put(&mut self, mut a: usize, mut b: usize, acc: Access) {
        self.cut(a, b);
        if let Some((&s, &(e, x))) = self.committed.range(..a).next_back() {
            if e == a && x == acc {
                self.committed.remove(&s);
                a = s;
            }
        }
        if let Some(&(e, x)) = self.committed.get(&b) {
            if x == acc {
                self.committed.remove(&b);
                b = e;
            }
        }
        self.committed.insert(a, (b, acc));
    }

    fn known(&self, at: usize) -> bool {
        self.known.range(..=at).next_back().is_some_and(|(_, &e)| at < e) || crate::gpu::uma::get().is_some_and(|u| u.is_sparse(at))
    }
}

fn write() -> std::sync::RwLockWriteGuard<'static, Map> {
    MAP.write().unwrap_or_else(|e| e.into_inner())
}

fn read() -> std::sync::RwLockReadGuard<'static, Map> {
    MAP.read().unwrap_or_else(|e| e.into_inner())
}

/// `[start, start + len)` is address space XCL reserved (or handed out) for the title: new
/// address space, to the end of its last page – what was recorded there is gone.
pub fn reserved(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = (start + len + 0xFFF) & !0xFFF;
    let mut g = write();
    // (a reservation recorded here went back to Windows unseen: all of it is gone)
    let stale: Vec<(usize, usize)> = g.known.range(..end).rev().take_while(|(_, &e)| e > start).map(|(&s, &e)| (s, e)).collect();
    for (s, e) in stale {
        g.known.remove(&s);
        g.cut(s, e);
    }
    g.cut(start & !0xFFF, end);
    g.known.insert(start, end);
}

/// The reservation at `start` is gone, with everything committed in it.
pub fn released(start: usize) {
    let mut g = write();
    if let Some(end) = g.known.remove(&start) {
        g.cut(start, end);
    }
}

/// What `take` forgot of a reservation: its span and what was committed in it.
pub struct Taken {
    start: usize,
    end: usize,
    committed: Vec<(usize, usize, Access)>,
}

/// `released`, keeping what is forgotten so that `restore` can put it back – for a release
/// that Windows may still refuse (None: no reservation was recorded at `start`).
pub fn take(start: usize) -> Option<Taken> {
    let mut g = write();
    let end = g.known.remove(&start)?;
    let committed = g.committed.range(..end).rev().take_while(|(_, &(e, _))| e > start).map(|(&s, &(e, acc))| (s.max(start), e.min(end), acc)).collect();
    g.cut(start, end);
    Some(Taken { start, end, committed })
}

/// The reservation `take` forgot is there after all, with what was committed in it.
pub fn restore(t: Taken) {
    let mut g = write();
    g.cut(t.start, t.end);
    g.known.insert(t.start, t.end);
    for (a, b, acc) in t.committed {
        g.put(a, b, acc);
    }
}

/// The reservation holding `at` (start, end), if XCL made it.
pub fn reservation(at: usize) -> Option<(usize, usize)> {
    read().known.range(..=at).next_back().filter(|(_, &e)| at < e).map(|(&s, &e)| (s, e))
}

/// `[start, start + len)` is committed with `protect`. Kept only where XCL sees the address
/// space go again: in a reservation it recorded, or in unified memory. (Elsewhere Windows is
/// asked: a span kept for memory released behind XCL would answer for whatever comes there next.)
pub fn committed(start: usize, len: usize, protect: u32) {
    if len == 0 {
        return;
    }
    let unified = crate::gpu::uma::get().is_some_and(|u| u.contains(start));
    let mut g = write();
    if unified || g.known.range(..=start).next_back().is_some_and(|(_, &e)| start < e) {
        g.put(start & !0xFFF, (start + len + 0xFFF) & !0xFFF, Access::of(protect));
    }
}

/// `[start, start + len)` is decommitted.
pub fn decommitted(start: usize, len: usize) {
    if len > 0 {
        write().cut(start & !0xFFF, (start + len + 0xFFF) & !0xFFF);
    }
}

/// The committed parts of `[start, start + len)` now have `protect`.
pub fn protected(start: usize, len: usize, protect: u32) {
    let (a, b) = (start & !0xFFF, (start + len + 0xFFF) & !0xFFF);
    let mut g = write();
    let parts: Vec<(usize, usize)> = g.committed.range(..b).rev().take_while(|(_, &(e, _))| e > a).map(|(&s, &(e, _))| (s.max(a), e.min(b))).collect();
    for (s, e) in parts {
        g.put(s, e, Access::of(protect));
    }
}

/// The least access over `[p, p + len)`, or None when part of it is outside what XCL recorded
/// (ask Windows).
pub fn access(p: usize, len: usize) -> Option<Access> {
    let end = p.checked_add(len.max(1))?;
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len.max(1))) {
        return Some(Access::Write);
    }
    let g = read();
    let mut least = Access::Write;
    let mut cur = p;
    while cur < end {
        match g.committed.range(..=cur).next_back() {
            Some((_, &(e, acc))) if e > cur => {
                least = least.min(acc);
                cur = e;
            }
            _ if g.known(cur) => {
                ANSWERED.fetch_add(1, Ordering::Relaxed);
                return Some(Access::None);
            }
            _ => {
                ASKED.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
    }
    ANSWERED.fetch_add(1, Ordering::Relaxed);
    Some(least)
}

/// The committed, readable runs (start, end) of `[p, p + len)`, or None when part of it is
/// outside what XCL recorded.
pub fn readable_runs(p: usize, len: usize) -> Option<Vec<(usize, usize)>> {
    let end = p.checked_add(len)?;
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(p, len)) {
        return Some(vec![(p, end)]);
    }
    let g = read();
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut cur = p;
    while cur < end {
        match g.committed.range(..=cur).next_back() {
            Some((_, &(e, acc))) if e > cur => {
                let hi = e.min(end);
                if acc >= Access::Read {
                    match out.last_mut() {
                        Some(last) if last.1 == cur => last.1 = hi,
                        _ => out.push((cur, hi)),
                    }
                }
                cur = hi;
            }
            _ if g.known(cur) => {
                // to the next committed run, or the end
                cur = g.committed.range(cur..end).next().map_or(end, |(&s, _)| s);
            }
            _ => return None,
        }
    }
    Some(out)
}

/// (answered from the record, asked of Windows) since the start
pub fn counts() -> (u64, u64) {
    (ANSWERED.load(Ordering::Relaxed), ASKED.load(Ordering::Relaxed))
}

// --- XCL's own reads and writes of the title's memory --------------------------------------

/// Copy `[src, src + len)` into `dst` without faulting if another thread unmaps or
/// decommits a page concurrently: `ReadProcessMemory` catches the fault in kernel mode and returns
/// `0`. Because Windows may return `read == 0` when a fault occurs later in the span, binary-split
/// failed multi-page ranges at 4 KiB boundaries so all still-committed pages are copied in O(log N)
/// calls and only unreadable pages are zeroed.
pub(crate) unsafe fn copy_committed_region(src: usize, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const PAGE: usize = 0x1000;

    unsafe fn copy_rec(proc: *mut c_void, src: usize, dst: *mut u8, len: usize) {
        if len == 0 {
            return;
        }
        let mut read = 0usize;
        if ReadProcessMemory(proc, src as *const c_void, dst as *mut c_void, len, &mut read) != 0 {
            return;
        }
        let read = read.min(len);
        let src = src + read;
        let dst = dst.add(read);
        let len = len - read;
        let page_end = (src & !(PAGE - 1)).saturating_add(PAGE).min(src + len);
        if page_end == src + len {
            std::ptr::write_bytes(dst, 0, len);
            return;
        }
        let mid = ((src + len / 2) & !(PAGE - 1)).max(page_end);
        let left = mid - src;
        copy_rec(proc, src, dst, left);
        copy_rec(proc, mid, dst.add(left), len - left);
    }

    copy_rec(GetCurrentProcess(), src, dst, len);
}

/// The committed, readable runs (start, end) of `len` bytes at `at`.
pub(crate) unsafe fn committed_runs(at: usize, len: usize) -> Vec<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let Some(limit) = at.checked_add(len) else { return Vec::new() };
    // (the coherent window is committed as a whole, and what XCL recorded needs no
    // `VirtualQuery`, which sizes a region by walking it)
    if let Some(runs) = readable_runs(at, len) {
        return runs;
    }
    let mut out = Vec::new();
    let mut p = at;
    while p < limit {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.RegionSize == 0 {
            break;
        }
        let end = (m.BaseAddress as usize).saturating_add(m.RegionSize).min(limit);
        if end <= p {
            break;
        }
        // R / RW / WC / X-R…, no guard
        if m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0 {
            out.push((p, end));
        }
        p = end;
    }
    out
}

/// The title's `out.len()` bytes at `at` into `out`, as far as they are there: what the title
/// unmaps, decommits or frees meanwhile – and what was never committed – is zeros. Windows makes
/// the copy and answers for memory that is gone; "readable" a moment ago is not readable now.
/// Every read XCL makes of the title's memory on its own schedule goes through here or through
/// `title_bytes` (a fault-safe copy, not one moment's picture of a batch of mappings).
pub fn copy_title(at: usize, out: &mut [u8]) {
    crate::hle::d3d12x::image_writeback::before_read(at, out.len());
    // SAFETY: Windows writes `out` only, and reads the title's memory for itself
    unsafe { copy_title_now(at, out) }
}

unsafe fn copy_title_now(at: usize, out: &mut [u8]) {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let len = out.len();
    if len == 0 {
        return;
    }
    if at.checked_add(len).is_none() {
        out.fill(0);
        return;
    }
    let mut read = 0;
    if ReadProcessMemory(GetCurrentProcess(), at as *const c_void, out.as_mut_ptr().cast(), len, &mut read) != 0 && read == len {
        return;
    }
    let mut pos = 0;
    for (lo, hi) in committed_runs(at, len) {
        out[pos..lo - at].fill(0);
        copy_committed_region(lo, out.as_mut_ptr().add(lo - at), hi - lo);
        pos = hi - at;
    }
    out[pos..].fill(0);
}

/// `len` bytes of the title's at `at`, all of them or none (None: some are not there – now).
pub fn title_bytes(at: usize, len: usize) -> Option<Vec<u8>> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    at.checked_add(len)?;
    let mut out = vec![0u8; len];
    let mut read = 0;
    // SAFETY: Windows writes `out` only, and reads the title's memory for itself
    let ok = len == 0 || unsafe { ReadProcessMemory(GetCurrentProcess(), at as *const c_void, out.as_mut_ptr().cast(), len, &mut read) } != 0 && read == len;
    ok.then_some(out)
}

/// `n` 32-bit words of the title's at `at` (`title_bytes`).
pub fn title_words(at: usize, n: usize) -> Option<Vec<u32>> {
    let bytes = title_bytes(at, n.checked_mul(4)?)?;
    Some(bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect())
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtWriteVirtualMemory(process: *mut c_void, base: *mut c_void, buffer: *const c_void, len: usize, written: *mut usize) -> i32;
}

/// `bytes` into the title's memory at `at`, written by Windows: false – and nothing more
/// written – where the title has unmapped, decommitted or freed it meanwhile. (Pages committed
/// on first use are committed first: Windows does not come by XCL's handler for them.)
///
/// # Safety
/// `[at, at + bytes.len())` is the title's memory, which XCL may write.
pub unsafe fn write_title(at: usize, bytes: &[u8]) -> bool {
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if bytes.is_empty() {
        return true;
    }
    crate::hle::kernelx::lazy::ensure(at, bytes.len());
    let mut written = 0;
    NtWriteVirtualMemory(GetCurrentProcess(), at as *mut c_void, bytes.as_ptr().cast(), bytes.len(), &mut written) >= 0 && written == bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reservation taken out for a release and put back (Windows refused) is as it was: its
    /// span and what is committed in it, a run shared with the reservation before it included.
    #[test]
    fn a_reservation_taken_and_restored_keeps_what_is_committed() {
        let (a, b) = (0x7E10_0000_0000usize, 0x7E10_0001_0000usize);
        reserved(a, 0x1_0000);
        reserved(b, 0x1_0000);
        committed(a + 0xF000, 0x1000, PAGE_READWRITE);
        committed(b, 0x3000, PAGE_READWRITE);
        committed(b + 0x8000, 0x1000, PAGE_READWRITE);
        let before = (access(b, 0x3000), access(b + 0x3000, 0x1000), access(b + 0x8000, 0x1000), readable_runs(a, 0x2_0000));
        assert_eq!(before.0, Some(Access::Write));
        assert_eq!(before.1, Some(Access::None));
        let taken = take(b).unwrap();
        assert_eq!(reservation(b), None);
        assert_eq!(access(b, 0x1000), None, "forgotten: Windows is asked");
        assert_eq!(access(a + 0xF000, 0x1000), Some(Access::Write), "the reservation before it keeps its part of the run");
        restore(taken);
        assert_eq!(reservation(b), Some((b, b + 0x1_0000)));
        assert_eq!((access(b, 0x3000), access(b + 0x3000, 0x1000), access(b + 0x8000, 0x1000), readable_runs(a, 0x2_0000)), before);
        assert!(take(b + 0x1000).is_none(), "only a reservation's start");
        released(a);
        released(b);
    }

    #[test]
    fn commits_merge_split_and_answer() {
        let base = 0x7000_0000_0000usize;
        reserved(base, 0x10_0000);
        assert_eq!(access(base, 0x1000), Some(Access::None), "reserved, not committed");
        committed(base + 0x1000, 0x3000, PAGE_READWRITE);
        committed(base + 0x4000, 0x1000, PAGE_READWRITE);
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::Write));
        protected(base + 0x2000, 0x1000, 2); // PAGE_READONLY
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::Read));
        decommitted(base + 0x3000, 0x1000);
        assert_eq!(access(base + 0x1000, 0x4000), Some(Access::None));
        assert_eq!(readable_runs(base, 0x6000), Some(vec![(base + 0x1000, base + 0x3000), (base + 0x4000, base + 0x5000)]));
        assert_eq!(access(base + 0x20_0000, 4), None, "outside the record");
        released(base);
        assert_eq!(access(base + 0x1000, 4), None);
    }

    /// A commit outside every reservation XCL recorded is not kept: XCL would not see that
    /// memory released.
    #[test]
    fn a_commit_outside_every_reservation_is_asked_of_windows() {
        let base = 0x7100_0000_0000usize;
        committed(base, 0x2000, PAGE_READWRITE);
        assert_eq!(access(base, 4), None);
        reserved(base, 0x10_0000);
        committed(base + 0x10_0000, 0x1000, PAGE_READWRITE);
        assert_eq!(access(base + 0x10_0000, 4), None, "past the reservation's end");
        released(base);
    }

    /// Address space reserved again starts with nothing committed, and the reservation recorded
    /// there before (released without XCL seeing it) is gone.
    #[test]
    fn a_new_reservation_starts_with_nothing_committed() {
        let base = 0x7200_0000_0000usize;
        reserved(base, 0x10_0000);
        committed(base, 0x10_0000, PAGE_READWRITE);
        // (released behind XCL, then handed out again in two pieces)
        reserved(base + 0x4_0000, 0x2_0000);
        assert_eq!(access(base + 0x4_0000, 4), Some(Access::None));
        assert_eq!(reservation(base + 0x4_0000), Some((base + 0x4_0000, base + 0x6_0000)));
        assert_eq!(reservation(base), None, "the old reservation is not there any more");
        assert_eq!(access(base + 0x8_0000, 4), None, "nor what was committed in the rest of it");
        released(base + 0x4_0000);
    }

    /// A reservation ends with its last page, as its commits do: releasing it leaves nothing.
    #[test]
    fn a_reservation_of_part_of_a_page_is_released_whole() {
        let base = 0x7300_0000_0000usize;
        reserved(base, 0x1800);
        committed(base, 0x1800, PAGE_READWRITE);
        assert_eq!(access(base + 0x1900, 4), Some(Access::Write));
        released(base);
        assert_eq!(access(base + 0x1900, 4), None);
    }
}
