//! What the children of one `xcl warm-up --blocking` pass (`warm_children`) have in flight,
//! for each other's governor (`real::DRIVER_COMPILES`): a table in named shared memory with one
//! slot per child (the compiles it runs, the giants among them, their expected peaks), and a
//! named mutex every admission takes while it reads the others and publishes its own. Admitting
//! against the free RAM under that one lock keeps two children from both taking the same memory.
//!
//! The counts are the table's rather than a named semaphore's: a semaphore's count is lost
//! when the child holding it is killed, while a dead child's slot is cleared by the parent
//! (`Held::clear`) as soon as it has exited. A child killed while it holds the mutex leaves it
//! abandoned, which Windows hands to the next waiter as owned.

use std::marker::PhantomData;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE};
use windows_sys::Win32::System::Threading::{CreateMutexW, OpenMutexW, ReleaseMutex, WaitForSingleObject, INFINITE, MUTEX_ALL_ACCESS};

/// Children of one pass at once at most (one slot each).
pub(super) const SLOTS: usize = 128;
/// `<parent pid>:<slot>`: set by the parent for each child (`join_from_env`).
pub(super) const ENV: &str = "XCL_WARM_PEERS";

/// Driver compiles in flight (`real::GovState`): running, giants among them, their expected peaks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Usage {
    pub running: usize,
    pub giants: usize,
    pub reserved: u64,
}

impl Usage {
    fn add(self, other: Usage) -> Usage {
        Usage { running: self.running + other.running, giants: self.giants + other.giants, reserved: self.reserved.saturating_add(other.reserved) }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Slot {
    running: u32,
    giants: u32,
    reserved: u64,
}

impl Slot {
    fn usage(self) -> Usage {
        Usage { running: self.running as usize, giants: self.giants as usize, reserved: self.reserved }
    }
}

/// The table of one pass: the parent's (`create`, no slot of its own) or a child's (`open`).
pub(super) struct Peers {
    lock: HANDLE,
    mapping: HANDLE,
    table: *mut Slot,
    mine: Option<usize>,
}

unsafe impl Send for Peers {}
unsafe impl Sync for Peers {}

fn name(parent: u32, what: &str) -> Vec<u16> {
    format!("Local\\xcl-warm-{parent}-{what}").encode_utf16().chain(Some(0)).collect()
}

impl Peers {
    /// The parent's table for pass `parent` (its pid), every slot empty.
    pub(super) fn create(parent: u32) -> std::io::Result<Peers> {
        let size = (SLOTS * std::mem::size_of::<Slot>()) as u32;
        unsafe {
            let mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, size, name(parent, "table").as_ptr());
            if mapping.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let lock = CreateMutexW(std::ptr::null(), 0, name(parent, "lock").as_ptr());
            if lock.is_null() {
                let e = std::io::Error::last_os_error();
                CloseHandle(mapping);
                return Err(e);
            }
            let peers = Self::map(lock, mapping, None)?;
            let held = peers.lock();
            for slot in 0..SLOTS {
                held.clear(slot);
            }
            drop(held);
            Ok(peers)
        }
    }

    /// Child `slot`'s view of the table of pass `parent`.
    pub(super) fn open(parent: u32, slot: usize) -> std::io::Result<Peers> {
        if slot >= SLOTS {
            return Err(std::io::Error::other(format!("warm-up peer slot {slot} out of range")));
        }
        unsafe {
            let mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name(parent, "table").as_ptr());
            if mapping.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let lock = OpenMutexW(MUTEX_ALL_ACCESS, 0, name(parent, "lock").as_ptr());
            if lock.is_null() {
                let e = std::io::Error::last_os_error();
                CloseHandle(mapping);
                return Err(e);
            }
            Self::map(lock, mapping, Some(slot))
        }
    }

    unsafe fn map(lock: HANDLE, mapping: HANDLE, mine: Option<usize>) -> std::io::Result<Peers> {
        let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, SLOTS * std::mem::size_of::<Slot>());
        if view.Value.is_null() {
            let e = std::io::Error::last_os_error();
            CloseHandle(lock);
            CloseHandle(mapping);
            return Err(e);
        }
        Ok(Peers { lock, mapping, table: view.Value.cast(), mine })
    }

    /// This child's slot (none for the parent's table).
    pub(super) fn own_slot(&self) -> Option<usize> {
        self.mine
    }

    /// The table, for this thread alone until the guard drops. An abandoned mutex (its owner died
    /// holding it) is owned all the same.
    pub(super) fn lock(&self) -> Held<'_> {
        unsafe { WaitForSingleObject(self.lock, INFINITE) };
        Held { peers: self, _thread: PhantomData }
    }
}

impl Drop for Peers {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.table.cast() });
            CloseHandle(self.mapping);
            CloseHandle(self.lock);
        }
    }
}

/// The table while its mutex is held (a Windows mutex belongs to the thread that took it).
pub(super) struct Held<'a> {
    peers: &'a Peers,
    _thread: PhantomData<*const ()>,
}

impl Held<'_> {
    fn slot(&self, i: usize) -> Slot {
        unsafe { self.peers.table.add(i).read_volatile() }
    }

    /// Every slot but this process's own.
    pub(super) fn others(&self) -> Usage {
        (0..SLOTS).filter(|&i| Some(i) != self.peers.mine).map(|i| self.slot(i).usage()).fold(Usage::default(), Usage::add)
    }

    /// Every slot.
    pub(super) fn total(&self) -> Usage {
        (0..SLOTS).map(|i| self.slot(i).usage()).fold(Usage::default(), Usage::add)
    }

    /// This child's own compiles in flight now (nothing for the parent).
    pub(super) fn publish(&self, usage: Usage) {
        if let Some(i) = self.peers.mine {
            let slot = Slot { running: usage.running as u32, giants: usage.giants as u32, reserved: usage.reserved };
            unsafe { self.peers.table.add(i).write_volatile(slot) };
        }
    }

    /// Slot `i` empty: its child has exited.
    pub(super) fn clear(&self, i: usize) {
        if i < SLOTS {
            unsafe { self.peers.table.add(i).write_volatile(Slot::default()) };
        }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        unsafe { ReleaseMutex(self.peers.lock) };
    }
}

static JOINED: OnceLock<Peers> = OnceLock::new();

/// A warm-up child joins its pass's table (`ENV`); without `ENV`, nothing to join.
pub(super) fn join_from_env() -> Result<(), String> {
    let Ok(value) = std::env::var(ENV) else { return Ok(()) };
    let (parent, slot) = value
        .split_once(':')
        .and_then(|(parent, slot)| Some((parent.parse::<u32>().ok()?, slot.parse::<usize>().ok()?)))
        .ok_or_else(|| format!("{ENV}={value:?} is not <parent pid>:<slot>"))?;
    let peers = Peers::open(parent, slot).map_err(|e| format!("cannot open the governor table of warm-up {parent}: {e}"))?;
    let _ = JOINED.set(peers);
    Ok(())
}

/// The table this process publishes its compiles in: a warm-up child's, else none.
pub(super) fn joined() -> Option<&'static Peers> {
    JOINED.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_child_sees_the_others_and_the_parent_sees_all() {
        let parent = std::process::id() ^ 0x5a5a_0000;
        let table = Peers::create(parent).unwrap();
        let (a, b) = (Peers::open(parent, 1).unwrap(), Peers::open(parent, 7).unwrap());
        assert!(Peers::open(parent, SLOTS).is_err());
        a.lock().publish(Usage { running: 2, giants: 1, reserved: 3 << 30 });
        b.lock().publish(Usage { running: 1, giants: 0, reserved: 1 << 30 });
        assert_eq!(a.lock().others(), Usage { running: 1, giants: 0, reserved: 1 << 30 });
        assert_eq!(b.lock().others(), Usage { running: 2, giants: 1, reserved: 3 << 30 });
        assert_eq!(table.lock().total(), Usage { running: 3, giants: 1, reserved: 4 << 30 });
        assert_eq!(table.lock().others(), table.lock().total(), "the parent has no slot of its own");
        table.lock().publish(Usage { running: 9, giants: 9, reserved: 9 });
        assert_eq!(table.lock().total().running, 3, "the parent publishes nothing");
        table.lock().clear(1);
        assert_eq!(b.lock().others(), Usage::default(), "a dead child's slot is cleared by the parent");
        let nested = (a.lock(), b.lock());
        drop(nested);
        assert!(Peers::open(parent ^ 1, 1).is_err(), "no table of another pass");
    }
}
