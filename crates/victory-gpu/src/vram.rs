//! GPU-owned textures in title memory, aliased as on the console via 64 KiB
//! tiled video memory pages and exact-size fallback arenas.
//!
//! Ported from XCL (`crates/xcl/src/hle/d3d12x/vram.rs`).
//! A console title places render targets, depth buffers, and UAV textures at
//! addresses of its own memory and reuses/aliases that memory. Creating an
//! independent `CreateCommittedResource` per placement breaks aliasing and
//! exhausts WDDM commit / VRAM (`DXGI_ERROR_DEVICE_HUNG`).
//!
//! `Pages` allocates each texture as a reserved (tiled) resource whose 64 KiB
//! tiles map onto a shared pool of 64 KiB VRAM pages keyed by the title's
//! 64 KiB page address. Resources that cannot be reserved (e.g. D32S8, MSAA,
//! or unaligned addresses) fall back to exact-size `Arenas`.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

const PAGE: usize = 64 << 10;
const POOL_PAGES: u32 = 256; // 16 MiB per pool heap
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
const COOL: std::time::Duration = std::time::Duration::from_millis(500);

/// Returns `true` for GPU-owned resources (RenderTarget, DepthStencil, UAV).
pub(crate) fn gpu_owned(flags: u32) -> bool {
    const RT_DS_UAV: u32 = 0x1 | 0x2 | 0x4;
    flags & RT_DS_UAV != 0
}

struct Arena<H> {
    base: usize,
    size: usize,
    align: usize,
    heap: H,
    users: u32,
    idle_since: Option<std::time::Instant>,
    id: u64,
}

struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas {
            list: Vec::new(),
            next_id: 0,
        }
    }

    fn sweep(&mut self, now: std::time::Instant) {
        self.list
            .retain(|a| a.users > 0 || a.idle_since.is_none_or(|t| now.duration_since(t) < IDLE));
    }

    fn take(
        &mut self,
        va: usize,
        bytes: usize,
        align: usize,
        now: std::time::Instant,
        make: impl FnOnce(usize, usize) -> Option<H>,
    ) -> Option<(H, usize, u64)> {
        self.sweep(now);
        let fits = |a: &Arena<H>| {
            a.align >= align
                && a.base <= va
                && va + bytes <= a.base + a.size
                && (va - a.base).is_multiple_of(align)
        };
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let heap = make(size, align)?;
                self.next_id += 1;
                self.list.push(Arena {
                    base: va,
                    size,
                    align,
                    heap,
                    users: 0,
                    idle_since: None,
                    id: self.next_id,
                });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        Some((a.heap.clone(), va - a.base, a.id))
    }

    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize {
        self.list.iter().map(|a| a.size).sum()
    }
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
static ARENA_OF: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Returns `(bytes in active arenas, total arena bytes ever allocated)`.
pub fn arena_bytes() -> (usize, usize) {
    (
        ARENAS.lock().unwrap_or_else(|e| e.into_inner()).bytes(),
        MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed),
    )
}

fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: valid device and heap description.
    if let Err(e) = unsafe { dev.CreateHeap(&desc, &mut heap) } {
        tracing::warn!(va = format_args!("{va:#x}"), size, error = %e, "vram arena CreateHeap failed");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    tracing::debug!(
        va = format_args!("{va:#x}"),
        size,
        made,
        "vram arena allocated"
    );
    heap
}

type Tile = (u32, u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

struct Page {
    tile: Tile,
    users: u32,
    idle_since: Option<std::time::Instant>,
}

struct Target {
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
}

struct Pages {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    free: Vec<Tile>,
    cooling: VecDeque<(Tile, std::time::Instant)>,
    pages: HashMap<usize, Page>,
    extra_pages: HashMap<(usize, usize), Page>,
    idle: VecDeque<(PageKey, std::time::Instant)>,
    targets: HashMap<usize, Target>,
    #[allow(dead_code)]
    waited: HashMap<usize, u64>,
}

// SAFETY: D3D12 objects are free-threaded and protected by `PAGES` mutex.
unsafe impl Send for Pages {}

static PAGES: Mutex<BTreeMap<usize, Option<Pages>>> = Mutex::new(BTreeMap::new());

fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let ours = h == heap;
        let offset = if ours { t } else { 0 };
        match out.last_mut() {
            Some(last) if last.0 == ours && (!ours || last.1 + last.2 == offset) => last.2 += 1,
            _ => out.push((ours, offset, 1)),
        }
    }
    out
}

fn sweep_idle(
    idle: &mut VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut HashMap<usize, Page>,
    extra_pages: &mut HashMap<(usize, usize), Page>,
    cooling: &mut VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while let Some(&(key, since)) = idle.front() {
        if now.duration_since(since) < IDLE {
            break;
        }
        idle.pop_front();
        let gone = match key {
            PageKey::Own(addr) => {
                if pages.get(&addr).is_some_and(|p| p.users == 0 && p.idle_since == Some(since)) {
                    pages.remove(&addr).map(|p| p.tile)
                } else {
                    None
                }
            }
            PageKey::Extra(va, tile) => {
                if extra_pages
                    .get(&(va, tile))
                    .is_some_and(|p| p.users == 0 && p.idle_since == Some(since))
                {
                    extra_pages.remove(&(va, tile)).map(|p| p.tile)
                } else {
                    None
                }
            }
        };
        if let Some(tile) = gone {
            cooling.push_back((tile, now));
        }
    }
}

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: `o` matches `D3D12_FEATURE_D3D12_OPTIONS`.
        unsafe {
            dev.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut o).cast(),
                std::mem::size_of_val(&o) as u32,
            )
            .ok()?;
        }
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            tracing::info!(
                tier = o.TiledResourcesTier.0,
                "tiled resources tier < 2; using arenas"
            );
            return None;
        }
        // SAFETY: creating an internal direct queue and fence for tile mappings.
        let queue: ID3D12CommandQueue = unsafe {
            dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .ok()?;
        // SAFETY: valid fence creation.
        let fence: ID3D12Fence = unsafe { dev.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
        tracing::info!("VRAM 64 KiB tiled resource paging enabled");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            free: Vec::new(),
            cooling: VecDeque::new(),
            pages: HashMap::new(),
            extra_pages: HashMap::new(),
            idle: VecDeque::new(),
            targets: HashMap::new(),
            waited: HashMap::new(),
        })
    }

    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(
            &mut self.idle,
            &mut self.pages,
            &mut self.extra_pages,
            &mut self.cooling,
            now,
        );
    }

    unsafe fn take_tile(&mut self, dev: &ID3D12Device) -> Option<Tile> {
        if let Some(t) = self.free.pop() {
            return Some(t);
        }
        if self.cooling.front().is_some_and(|(_, at)| at.elapsed() >= COOL) {
            return self.cooling.pop_front().map(|(t, _)| t);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: valid device and heap desc.
        if let Err(e) = unsafe { dev.CreateHeap(&desc, &mut heap) } {
            tracing::warn!(error = %e, "no video memory for more 64 KiB texture pages");
            return None;
        }
        self.pools.push(heap?);
        let h = self.pools.len() as u32 - 1;
        self.free.extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        let mut heaps: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges
                .iter()
                .map(|r| {
                    if r.0 {
                        D3D12_TILE_RANGE_FLAG_NONE
                    } else {
                        D3D12_TILE_RANGE_FLAG_SKIP
                    }
                })
                .collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE {
                NumTiles: tiles.len() as u32,
                UseBox: false.into(),
                ..Default::default()
            };
            // SAFETY: `res` and `self.pools[h]` are valid live D3D12 objects and
            // `flags`, `offsets`, `counts` have matching length `ranges.len()`.
            unsafe {
                self.queue.UpdateTileMappings(
                    res,
                    1,
                    Some(&coord),
                    Some(&region),
                    &self.pools[h as usize],
                    ranges.len() as u32,
                    Some(flags.as_ptr()),
                    Some(offsets.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        self.value += 1;
        // SAFETY: `self.fence` is valid.
        let _ = unsafe { self.queue.Signal(&self.fence, self.value) };
    }

    unsafe fn bind(
        &mut self,
        dev: &ID3D12Device,
        res: &ID3D12Resource,
        va: usize,
        console_bytes: Option<u64>,
        total: u32,
    ) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let now = std::time::Instant::now();
        self.sweep(now);
        if total == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let owned = match console_bytes {
            Some(cb) => ((va as u64 + cb).div_ceil(PAGE as u64) - (va as u64 / PAGE as u64)) as u32,
            None => total,
        }
        .min(total);
        let base_page = (va / PAGE) * PAGE;
        let mut tiles = Vec::with_capacity(total as usize);
        let mut used = Vec::with_capacity(owned as usize);
        let mut used_extra = Vec::new();

        for i in 0..owned {
            let addr = base_page + (i as usize) * PAGE;
            let tile = if let Some(p) = self.pages.get_mut(&addr) {
                p.users += 1;
                p.idle_since = None;
                p.tile
            } else {
                // SAFETY: `dev` is a valid D3D12 device.
                let Some(t) = (unsafe { self.take_tile(dev) }) else {
                    self.release_tiles(used, used_extra, now, false);
                    return None;
                };
                self.pages.insert(
                    addr,
                    Page {
                        tile: t,
                        users: 1,
                        idle_since: None,
                    },
                );
                t
            };
            tiles.push(tile);
            used.push(addr);
        }

        for i in owned..total {
            let key = (va, i as usize);
            let tile = if let Some(p) = self.extra_pages.get_mut(&key) {
                p.users += 1;
                p.idle_since = None;
                p.tile
            } else {
                // SAFETY: `dev` is a valid D3D12 device.
                let Some(t) = (unsafe { self.take_tile(dev) }) else {
                    self.release_tiles(used, used_extra, now, false);
                    return None;
                };
                self.extra_pages.insert(
                    key,
                    Page {
                        tile: t,
                        users: 1,
                        idle_since: None,
                    },
                );
                t
            };
            tiles.push(tile);
            used_extra.push(key);
        }

        // SAFETY: `res` is a reserved resource of `total` tiles.
        unsafe { self.map(res, &tiles) };
        Some((used, used_extra))
    }

    fn release_tiles(
        &mut self,
        used: Vec<usize>,
        used_extra: Vec<(usize, usize)>,
        now: std::time::Instant,
        fast_extra: bool,
    ) {
        for addr in used {
            if let Some(p) = self.pages.get_mut(&addr) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    p.idle_since = Some(now);
                    self.idle.push_back((PageKey::Own(addr), now));
                }
            }
        }
        for key in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&key) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if fast_extra {
                        let t = p.tile;
                        self.extra_pages.remove(&key);
                        self.cooling.push_back((t, now));
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(key.0, key.1), now));
                    }
                }
            }
        }
    }
}

fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: `o` matches `D3D12_FEATURE_D3D12_OPTIONS`.
    unsafe {
        dev.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut o).cast(),
            std::mem::size_of_val(&o) as u32,
        )
        .is_ok()
            && o.ResourceHeapTier.0 >= 2
    }
}

fn can_reserve(support2: u32, samples: u32, quality_levels: u32) -> bool {
    const TILED: u32 = D3D12_FORMAT_SUPPORT2_TILED.0 as u32;
    support2 & TILED != 0 && (samples <= 1 || quality_levels > 0)
}

fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: desc.Format,
        ..Default::default()
    };
    // SAFETY: `f` matches `D3D12_FEATURE_FORMAT_SUPPORT`.
    if unsafe {
        dev.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut f).cast(),
            std::mem::size_of_val(&f) as u32,
        )
    }
    .is_err()
    {
        return true;
    }
    if !can_reserve(f.Support2.0 as u32, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
        Format: desc.Format,
        SampleCount: samples,
        Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE,
        NumQualityLevels: 0,
    };
    if samples > 1
        // SAFETY: `m` matches `D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS`.
        && unsafe {
            dev.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut m).cast(),
                std::mem::size_of_val(&m) as u32,
            )
        }
        .is_err()
    {
        return true;
    }
    can_reserve(f.Support2.0 as u32, samples, m.NumQualityLevels)
}

unsafe fn place_on_pages(
    dev: &ID3D12Device,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: Option<&D3D12_CLEAR_VALUE>,
) -> Option<ID3D12Resource> {
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN
        || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D
        || !va.is_multiple_of(PAGE)
    {
        return None;
    }
    let tiled = D3D12_RESOURCE_DESC {
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        ..*desc
    };
    if !reservable(dev, &tiled) {
        return None;
    }
    let mut g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let pages = g
        .entry(dev.as_raw() as usize)
        // SAFETY: `dev` is a valid D3D12 device.
        .or_insert_with(|| unsafe { Pages::new(dev) })
        .as_mut()?;

    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: `tiled` is a valid reserved resource description.
    let made = unsafe {
        dev.CreateReservedResource(
            &tiled,
            D3D12_RESOURCE_STATES(state as i32),
            clear.map(|c| c as *const D3D12_CLEAR_VALUE),
            &mut res,
        )
    };
    if made.is_err() {
        return None;
    }
    let res = res?;
    let mut total = 0u32;
    // SAFETY: `res` is a live reserved resource.
    unsafe {
        dev.GetResourceTiling(&res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    }
    // SAFETY: `dev` and `res` are valid.
    let (used, used_extra) = unsafe { pages.bind(dev, &res, va, None, total) }?;
    let raw = res.as_raw() as usize;
    pages.targets.insert(
        raw,
        Target {
            mapped: true,
            used,
            used_extra,
        },
    );
    drop(g);
    PLACED_TEXTURES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert(raw);
    Some(res)
}

/// Places a GPU-owned texture at title virtual address `va` using 64 KiB
/// shared tiled pages (`Pages`), or exact-size `Arenas` if reserved placement
/// is unavailable. Returns `None` if neither applies.
pub(crate) unsafe fn place(
    dev: &ID3D12Device,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: Option<&D3D12_CLEAR_VALUE>,
) -> Option<ID3D12Resource> {
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("VICTORY_VRAM_ARENAS").is_none()
        // SAFETY: forwarded from caller.
        && let Some(res) = unsafe { place_on_pages(dev, va, desc, state, clear) }
    {
        return Some(res);
    }
    // SAFETY: `desc` is a valid resource description.
    let info = unsafe { dev.GetResourceAllocationInfo(0, &[*desc]) };
    if info.SizeInBytes == u64::MAX
        || info.SizeInBytes == 0
        || !(va as u64).is_multiple_of(info.Alignment.max(1))
    {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let (heap, offset, id) = ARENAS.lock().unwrap_or_else(|e| e.into_inner()).take(
        va,
        info.SizeInBytes as usize,
        align,
        now,
        |size, align| arena_heap(dev, va, size, align),
    )?;
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: `heap` and `offset` were verified to fit `info.SizeInBytes`.
    let placed = unsafe {
        dev.CreatePlacedResource(
            &heap,
            offset as u64,
            desc,
            D3D12_RESOURCE_STATES(state as i32),
            clear.map(|c| c as *const D3D12_CLEAR_VALUE),
            &mut res,
        )
    };
    if placed.is_err() || res.is_none() {
        ARENAS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(id, now);
        return None;
    }
    let res = res?;
    let raw = res.as_raw() as usize;
    PLACED_TEXTURES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert(raw);
    ARENA_OF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(raw, id);
    Some(res)
}

/// Releases tiles or arena references when a host texture is dropped.
pub(crate) fn forget(resource: usize) {
    if let Some(set) = PLACED_TEXTURES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        set.remove(&resource);
    }
    if let Some(id) = ARENA_OF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|m| m.remove(&resource))
    {
        let now = std::time::Instant::now();
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        arenas.release(id, now);
        arenas.sweep(now);
    }
    let mut g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else {
            continue;
        };
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        return;
    }
}

/// Ensures `queue` waits on any pending `UpdateTileMappings` fence before
/// executing command lists.
#[allow(dead_code)]
pub(crate) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    // SAFETY: `queue` is a valid command queue.
    if unsafe { queue.GetDevice(&mut dev) }.is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let mut g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else {
        return;
    };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        // SAFETY: `queue` and `pages.fence` belong to the same device.
        let _ = unsafe { queue.Wait(&pages.fence, pages.value) };
        pages.waited.insert(key, pages.value);
    }
}

/// Returns `(active page bytes, total pool heap bytes)`.
pub fn page_bytes() -> (usize, usize) {
    let g = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    g.values().flatten().fold((0, 0), |(used, pools), p| {
        (
            used + (p.pages.len() + p.extra_pages.len()) * PAGE,
            pools + p.pools.len() * POOL_PAGES as usize * PAGE,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(
            heap_ranges(0, &tiles),
            [(true, 5, 2), (false, 0, 2), (true, 9, 3)]
        );
        assert_eq!(
            heap_ranges(1, &tiles),
            [(false, 0, 2), (true, 0, 2), (false, 0, 3)]
        );
        assert_eq!(
            heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]),
            [(true, 3, 1), (true, 1, 2)]
        );
    }

    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..10usize)
            .map(|i| {
                (
                    i * PAGE,
                    Page {
                        tile: (0, i as u32),
                        users: 1,
                        idle_since: None,
                    },
                )
            })
            .collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([(
            (7 * PAGE, 3),
            Page {
                tile: (1, 0),
                users: 1,
                idle_since: None,
            },
        )]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();

        fn release(
            p: &mut Page,
            idle: &mut VecDeque<(PageKey, Instant)>,
            key: PageKey,
            when: Instant,
        ) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }

        for a in [PAGE, 2 * PAGE] {
            release(
                pages.get_mut(&a).unwrap(),
                &mut idle,
                PageKey::Own(a),
                at(0),
            );
        }
        release(
            extra.get_mut(&(7 * PAGE, 3)).unwrap(),
            &mut idle,
            PageKey::Extra(7 * PAGE, 3),
            at(100),
        );
        {
            let p2 = pages.get_mut(&(2 * PAGE)).unwrap();
            (p2.users, p2.idle_since) = (1, None);
        }

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty());
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)]);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(
            cooling.iter().map(|c| c.0).collect::<Vec<_>>(),
            [(0, 1), (1, 0)]
        );
        assert!(pages.contains_key(&(2 * PAGE)));
    }
}
