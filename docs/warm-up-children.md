# xcl warm-up --blocking in child processes

ZOR-602, CGBJ audit item 5. `xcl warm-up --blocking` now runs its compiles in child processes of
the same xcl.exe. `--in-process` (or `XCL_WARM_CHILDREN=0`) keeps the old single-process pass.

## Why

In one process, a pass has two problems:

- Memory that DXC and the NVIDIA driver take for a giant (1 MB+ of DXIL, about 1.4 GB in the
  driver) is not always given back while the process lives. The private commit grows over a pass.
- A driver crash or hang ends the whole pass, and the keys after it are not done.

A child process gives all of its memory back when it exits. A crash or a hang in a child costs
only the keys that the child had in flight.

## Shape

The parent creates no D3D12 device. It reads the blocking list, orders it with
`pso_manifest::blocking_order`, and starts its first child. Each child creates its device and
writes a hello line with the translation era and the driver id. The parent opens the progress
record (`blocking-progress.bin` and `.driver`) and the bad list for that driver only after the
first hello. Then it loads the records and sends keys.

Protocol (one line each, `src/hle/d3d12x/warm_children.rs`):

| Direction | Line |
|---|---|
| parent to child (stdin) | `keys <hex>...`, `end`, `stop` |
| child to parent (stdout) | `@xcl-warm hello <era> <driver>`, `start <key>`, `done <key> <kind> <ok> <refused> <hits> <cold> <rejected> <saved> <elapsed_us> <admitted_us> <build_ms> <guest> <library_hits> <library_misses> <library_stored> <library_refused>`, `giant <key> <dxil_len>`, `missing <key>`, `mem <commit> <peak_ws> <peak_commit>`, `bye <dxc stage summary>` |

A stdout line without the `@xcl-warm` tag goes to the parent's stderr. Children inherit stderr,
so their `[SHADER]` and governor lines are kept. A child that cannot write to stdout exits,
because its parent is gone. Each child starts suspended (`CREATE_SUSPENDED`), is put in a job
object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, and is then resumed, so it ends with the parent.
A job object that cannot be made or assigned stops the pass; `--in-process` does not need one.

A pool child reads `pipelines.bin` once (`Store::index`, key to entry offset) and each worker
decodes the key it takes, so the order-reading thread only queues keys.

Every child's hello must name the first child's translation era and driver. A child that names
another (for example a WARP fallback that says `unknown`) is killed, its keys go back to the
queue, and it is counted; three such children end the pass.

### Lanes

- **Pool lane.** A few long-lived children (`--pool-children`, default 2 on 8 or more logical
  cores, else 1). Each child runs the existing worker loop on `cores / pool children` threads with
  `XCL_DXC_THREADS` set to the same share. The parent keeps each child at its thread count plus
  one key given. A pool child is recycled after `--recycle-keys` keys (default 400), or when its
  `mem` line reports a private commit over `--recycle-mb` (default 8192 MB). Recycled means
  that it gets `end`, finishes the keys it has, and exits; a new child takes its place.
- **Single-key lane.** Children that build one key and exit, at most `--giant-children` at once
  (default half the logical cores; a giant's child starts only when its peak fits the RAM). The lane takes predicted giants and suspects. A key is a
  predicted giant when its console shaders are at least `GIANT_GUEST_BYTES` =
  `GIANT_DXIL.div_ceil(TRANSLATED_DXIL_PER_GUEST_BYTE)` = 43,691 B, or when its longest earlier
  build (`pso_manifest::slow_ms`) took `GIANT_BUILD_MS` = 60 s or more. A key that is only slow
  (`SLOW_MS`) stays in the pool; the target order puts it first. A single-key child gets 1 worker thread and
  `max(2, cores / (pool + single children))` DXC threads.
- **Giants found in the pool.** Pool children are started with `--leave-giants`; single-key
  children never are. In a pool child every worker sets `LEAVE_GIANTS`, so `driver_slot` refuses
  a cold driver compile of `GIANT_DXIL` (1 MiB) or more of DXIL at once (`Refused::Giant`),
  without waiting for memory. Each caller tells `driver_slot` whether the pipeline's driver blob
  is kept on disk (`driver_cache::kept`); a giant with a blob starts if the memory lets
  it now, and is handed off otherwise, so a pool worker never waits for a giant's memory. The giant's
  translation and DXC output are already in the on-disk DXIL cache. The child replies
  `giant <key> <dxil_len>` instead of `done`, and the parent takes that DXIL length for the
  key's expected peak (`Job::peak`, used by `giant_fits` and the reservation). The parent takes the key out of that child's
  keys in flight (not done, not a suspect, not fruitless), appends it to the single-key lane's
  giant queue, so giants keep the order they were found in, and prints
  `  <key>: a giant (driver compile of 1 MiB or more of DXIL): left for a single-key child`.
  Its single-key child redoes only the driver compile, and the key's per-key line comes from that
  child's `done`. The summary counts these keys as "giants handed to single-key children".
- **Giant admission in the parent.** A giant child starts only when `giant_fits` is true: a
  single-key child is free (`--giant-children`) and the estimated peak
  (`pso_sched::peak(translated_dxil(guest))`) fits in the available RAM less every child's
  reservation, the pending reservations and the 1 GiB floor. A giant child the parent started is
  pending while its key is in flight. The table does not say which child a slot belongs to, so
  the parent takes as many pending children as the table has giants for published, and reserves
  the largest peaks of the rest, plus `CHILD_BASELINE` (1 GiB, an estimate of the device, the
  shader memory windows and the pipeline index) for each pending child that has not said hello.
  Without this, every step until a child's first admission would spend the same memory again.
  If no compile runs anywhere and no giant child is pending, the giant also starts, and its own
  governor waits for memory. The pool lane gets no new keys only while the next giant waits for
  memory with no giant compiling in the table, and either the pool holds that
  memory (the giant would fit with the pool's reservations and its children's last reported
  commit gone) or the giant has waited `GIANT_PATIENCE` (5 minutes). The pool children then drain
  and give the giant room, until a giant starts. After twice `GIANT_PATIENCE` the pool children
  are told to end, so their memory outside compiles goes too. This is the cross-process form of
  "a waiting giant keeps smaller compiles from taking its memory". A giant child waiting in its
  own governor is not published across processes; the pool pause covers it, since it counts only
  the giants compiling in the table. A giant that waits while another giant compiles never
  pauses the pool. There is no count of giants at once (the user's rule,
  2026-11-26 15:22Z): memory and commit headroom, each less 1 GiB, are the only limits.

If the user sets `XCL_DXC_THREADS`, every child keeps that value.

### Cross-process governance

`warm_peers.rs` holds a table in named shared memory (`Local\xcl-warm-<parent pid>-table`) with
one slot per child: compiles running, giants among them, and their estimated peaks. A named mutex
(`...-lock`) protects it. In a child, `Governor::acquire_within` takes the mutex, adds the other
slots to its `Load` (giants and reserved bytes), runs `pso_sched::admit`, and on admission writes
its own slot before it releases the mutex. `release` writes the slot again. So:

- Every process's reservations count against the one available RAM figure, and every compile
  also needs the commit headroom (commit limit less charge) to hold the running peaks, its own
  and 1 GiB.
- RAM admission keeps the 1 GiB floor and waits rather than refuses, as before. Two children
  cannot both admit against the same free RAM, because reading the others and publishing the own
  reservation happen under one lock.
- The idle breaker counts the other children's running compiles, so a child with nothing running
  does not break the memory wait while others compile.

The counts are in the table, not a named semaphore. A semaphore count held by a child that
is killed is lost. A dead child's slot is cleared by the parent after the child exits. A mutex
abandoned by a killed child is given to the next waiter as owned.

`real::memory_waited()` is a per-thread total of time spent in `acquire_within`: driver slots,
translation slots and cross-process waits. A key's `build_ms` is its elapsed time less the
change in `memory_waited()` across it. The parent passes `build_ms` to
`pso_manifest::note_build_time`.

### Crashes and hangs

- Each key's hang clock stops while it waits for memory: the child writes `wait <key>` when a
  governor admission starts waiting and `go <key>` when it ends (`WaitClock` in
  `acquire_within`).
- A key is "in flight" from its `start` line (for a single-key child, from the spawn) until its
  `done`. Keys given but not started go back to the front of their queue when the child exits.
- After a crash or a hang kill, every key in flight is a suspect. Each suspect is tried again
  once, alone, in a single-key child. If that child crashes too, the key is bad: the summary counts
  it, its per-key line says `failed` with the reason appended, and it is written to
  `blocking-bad.txt` beside `blocking-progress.bin` as `<key> <era> <driver> <reason>`.
  Later passes with the same era and driver leave bad keys out. `--fresh` deletes the file.
- A key running for `--key-timeout-s` (default 1800 s; real DXC compiles reached 909 s), waits
  for memory left out, kills its child; this counts as a hang. A timeout never bad-lists a key:
  a key that hangs again alone is counted and left for the next pass (`deferred` on its line).
  Only a crash (the child dying without being killed) puts a key on the bad list. A child that says no hello within the timeout is also killed.
- A child that exits cleanly (`bye`, exit code 0) with keys still in flight was stopped. Its keys
  count as not started, not as suspects.
- Three children in a row that end without any key done (before hello, or crashing with no key in
  flight) end the pass with an error.

### Output

The per-key line, the `checked;` line, the start line (`... workers on ... logical cores; ...;
N DXC threads`) and the final summary keep their text. New fields are only at the end of each
line: the children and lanes on the start line, the pool and single-key child counts on the
`checked;` line, and on the summary the children started, recycled, crashed and hung, the suspects
retried, the keys marked bad, the bad keys left out, the children killed for another translation
or driver, and the giants handed to single-key children. The "peak working set" and "peak private
commit" values in child mode are the largest that any one child reported. The `warm-up shaders:`
line is the sum of each child's `stage_summary`, number by number.

## Defaults

| Flag | Default | Reason |
|---|---|---|
| `--pool-children` | 2 (1 under 8 cores) | Two pools let one recycle while the other works. |
| `--giant-children` | half the logical cores | Giants start by RAM; this bounds the processes. 1–96, and at most the peer table's 128 slots less twice the pool children and one. |
| `--recycle-keys` | 400 | Bounds the growth of a child between recycles without paying device start-up too often. |
| `--recycle-mb` | 8192 | Two thirds of the old in-process `WARM_UP_COMMIT` (12 GiB). |
| `--key-timeout-s` | 1800 | About twice the longest DXC compile seen (909 s). |

## Open risks

- The RAM accounting sees what children publish (the estimated peaks of admitted compiles) and
  the parent's pending reservations. A child's memory outside a compile (device, caches, leaked
  driver memory) is only estimated (`CHILD_BASELINE`) until its hello; after that the OS
  "available" figure includes it, but only as it happens.
- The in-process giant queue (`giant_waits`) and `starving` count are per process. A giant that
  waits in one child does not hold back small compiles in another child. The parent's pool gate
  covers this only while no giant compiles anywhere.
- A key's wait for memory reaches the parent as a `wait`/`go` line, so the hang clock may be a
  few milliseconds off around each wait.
- A giant predicted from console bytes can be wrong. A small-looking key that becomes a giant is
  found in a pool child only after its translation and DXC compile, and is then handed to the
  single-key lane. Each handed giant costs the start of a single-key child (device and
  `pipelines.bin`) before its driver compile, while a single-key child's place may stand idle.
- Each child creates its own D3D12 device, and each pool child keeps `pipelines.bin` in memory.

## CGBJ code

Adapted from CGBJ (VIctory) commit f4b48ad, `apps/victory/src/warm.rs`, Copyright (C) the
VIctory/CGBJ contributors, GPL-3.0-or-later. Each unit is marked in `warm_children.rs`:

- the reply line for each key (`made <n> <ms> <there>`) became `Msg::Done`;
- the hang kill (a watcher that kills a child not heard of for `LONGEST`) became the timeout for
  each key in `Coordinator::step`;
- `warm-bad.txt` (`<key> <why>`, left out of later passes) became `blocking-bad.txt` with era
  and driver;
- the child that exits when it cannot write to its parent (`Reply::send`);
- the child started from `current_exe()` with a piped stdout (`ProcessSpawner::spawn`).

## Pipeline library

Every translated pipeline a warm-up or the game builds is also stored in a D3D12 pipeline library
(`ID3D12PipelineLibrary`, `pso_library.rs`), and every build site tries the library first. The
behaviour follows VIctory/CGBJ `translate --cache --pipelines`; none of its code is used.

- **Name and folder.** A pipeline is stored under its driver blob's file name (`driver_cache`),
  which already hashes the host description and its DXIL. The files are in
  `<driver cache>\library\<translation era>\`, so another driver, adapter or translation uses
  another folder.
- **Load first.** Compute, graphics, pipeline-state stream and mesh builds (in the game and in the
  warm-up) call `LoadComputePipeline`, `LoadGraphicsPipeline` or `LoadPipeline` with the same
  description they build from, without a cached blob, after their driver slot and before the blob
  cache. D3D12 checks the description, so a hit is the pipeline the build would make. A miss or
  any error goes on to the blob cache and the cold compile as before. A pipeline that built goes
  into the library; one the title gets as a stand-in, or a stream rebuilt without its rect list GS,
  does not. The mesh warm-up (`MeshWarmBuild`) stores the pipelines it makes for their blobs.
- **Off by default, game only, keyed by description.** The library runs only with
  `XCL_PSO_LIBRARY=1`, and only in the game: a warm-up process makes none. Each entry is named
  by the translation era and a key that hashes the exact description it was built from: every
  shader's bytes, the root signature's static samplers, the memory windows and all state
  (`real::library_key`). Each shard has a key list, `library-<writer>.<generation>.keys`, with
  the era it was written under. A shard is read only when its list carries this build's era, and a
  load asks the driver only for a key a read shard lists. A shard of another build, or one without
  a key list (every shard written before this format), is skipped and kept as it is. The caches
  line counts hits, misses, stores, skipped shards and loads the driver refused.
- **Shards.** Each writer has its own files, `library-<writer>.<generation>.bin`: a warm-up child
  writes the shards of its warm-up peer slot (`w00` to `w127`, reused when a child exits), an
  in-process warm-up writes `w00`, and the game writes `game`. No two live processes write one
  file. At the start a process reads every shard in the folder, each through its own read-only
  library over a read-only file mapping that stays open as long as that library. Loads use only
  these. New pipelines go to the writer's own library, which nothing loads from: it is made
  over the writer's newest shard while that is under 32 MiB (`SEGMENT`), else made empty. So a
  save rewrites at most one segment plus what is new, and a writer has about one shard per
  32 MiB it has stored.
- **Saves.** A save runs on its own thread. It takes the writer's library from the stores (they
  go to a short list meanwhile) and serializes it with no lock held into a new temporary file of
  `GetSerializedSize` bytes through a mapped view, so no load and no title thread waits for it.
  The file is flushed and renamed to the writer's next generation. A rename never replaces a
  file: if the name is taken, the save takes the next free generation. The new file is read back
  when the writer adds to it or when it supersedes an older generation. The writer's library is
  then handed on: made over the new file while that is under a segment, else made empty, and the
  listed pipelines are stored in it. The generation the old library was made over is removed,
  because the new generation holds all of it; a full segment is never rewritten or removed. Each
  save logs the writer, the old and new generation, the serialized size, the new pipelines, the
  time since the last save, the time it took, and the files removed. The game and warm-ups save
  at most once every 5 minutes (`SAVE_INTERVAL`), only with at least 256 new pipelines
  (`SAVE_MIN`), and at the end: a child's `bye`, the end of an in-process warm-up, the
  process's exit. If a save fails, the library stays as it was and nothing is removed. The caches
  line counts the saves, the megabytes written, the longest save and the longest any load or
  store waited for the library's locks.
- **Exit save.** At a clean process exit, the save waits at most 200 ms in total for the
  libraries' locks. A library that is still busy (for example, a periodic save running) is
  skipped with a log line.
- **Refused shards.** If the driver refuses a shard with `D3D12_ERROR_DRIVER_VERSION_MISMATCH`,
  `D3D12_ERROR_ADAPTER_NOT_FOUND` or `E_INVALIDARG` (another driver or adapter, a damaged file),
  it is renamed to `.bin.invalid` (never over an earlier one; left in place if the rename fails).
  Any other failure (the file busy or unreadable, no memory) leaves the file exactly as it is
  and the shard is not read this run. Either way the pipelines in it are built again, one line
  is logged, and it never fails a build or a run.
- **Removal.** The only file ever removed is the generation a writer's library was made over,
  after its successor has been saved, flushed, renamed, reopened and accepted by the driver.
  Refused, unreadable and `.invalid` files, other writers' shards, and generations the library
  was not made over are never removed.
- **Threads.** D3D12 documents the library as thread-safe except for two threads loading the same
  pipeline, so loads of one name are serialized. A device's libraries are opened outside the
  device table's lock; if two threads race, the first one in the table is used.
- **Known gaps.** A folder of an earlier translation era is never read again and never deleted,
  so old eras stay on disk until the user removes them. An earlier build saved one generation
  per save, each holding only that save's pipelines. Those older generations of a writer are a
  one-off leftover: a library can't list its names, so they can't be folded into the newest
  generation. They stay read-only and are never removed. The same applies to a generation left
  behind when a process ends between a save's rename and its removal.
- **Counts.** The caches line and both warm-up summaries end with `pipeline library N hits,
  N misses, N stored, N shards refused`. Warm-up children send their counts on each `done` line.
- `XCL_PSO_LIBRARY=0` turns the library off. `XCL_PSO_DISK_CACHE=0` turns it off too, because
  the names come from the blob cache.
