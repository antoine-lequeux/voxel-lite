# voxel-lite

**voxel-lite** is a high-performance procedural voxel engine written from scratch in **C++23** and **Vulkan 1.3** (with shaders written in **Slang**). It is designed around the goal of rendering massive and detailed voxel worlds at hundreds of frames per second on modest hardware, with a very high render distance.
The project was built with **Data-Oriented Design** and idiomatic modern C++, to favor contiguous memory layouts and compile-time polymorphism over traditional heavy OOP.

The worlds are generated procedurally using **Perlin noise** and include continents, oceans, mountain ranges, forests, and underground cave networks, while keeping RAM and VRAM usage quite low.

### Performance & Memory Highlights

On a laptop with an entry-level GPU (2 Gigabytes VRAM), the engine achieves:
- **400+ FPS** at a render distance of **64 chunks (64 x 32m = 2 km view distance)**.
- **60+ FPS** at a render distance of **256 chunks (256 x 32m = 8 km view distance)**, and scales up to **512 chunks (16 km)**.

Reaching these numbers without running out of memory or choking the GPU required rethinking how voxels are stored, generated, culled, and drawn:
- **5-level octree-style chunk LODs**: distant chunks scale their voxel size from $1\text{ m}^3$ ($32^3$ blocks) down to $16\text{ m}^3$ ($2^3$ blocks), reducing voxel storage by up to **$4096\times$** in outer rings and drastically cutting the number of triangles sent to the GPU.
- **3D portal cave culling**: every chunk computes a 21-bit face-to-face 3D visibility mask. An asynchronous breadth-first search (BFS) only loads chunks reachable through open portals, leaving 99% of underground caves unloaded and unmeshed.
- **32-bit packed vertices & GPU-driven rendering**: every vertex is compressed into a single `uint32_t` (4 bytes). A GPU compute shader frustum-culls all active chunks and feeds a single `vkCmdDrawIndexedIndirectCount` call per frame.
- **Zero-overhead pools & `mimalloc`**: chunks live in pre-indexed flat slot pools ([`src/engine/containers.hpp`](src/engine/containers.hpp)) backed by Microsoft's [`mimalloc`](CMakeLists.txt#L106) allocator to eliminate heap fragmentation and allocator contention across worker threads.

---

## 1. Vulkan 1.3 Render Engine

The renderer ([`src/renderer/vulkancontext.hpp`](src/renderer/vulkancontext.hpp), [`src/renderer/vulkancontext.cpp`](src/renderer/vulkancontext.cpp)) uses modern **Vulkan 1.3** features (dynamic rendering, synchronization2, buffer device address (`VK_KHR_buffer_device_address`), and descriptor indexing) to keep CPU draw overhead near zero regardless of how many thousands of chunks are loaded.

### Pointer-Based Vertex Pulling (No Vertex Buffers Bound)
Instead of binding traditional vertex attributes or recording one draw call per chunk, every chunk mesh uploads its vertices to a GPU storage buffer and registers its **64-bit GPU memory address** (`device_address`) and quad count in a global [`ChunkDrawTable`](src/engine/chunk.hpp#L77-L114).
- Only the modified slots in `ChunkDrawTable` are scatter-uploaded to the GPU each frame ([`ChunkDrawTable::flush_dirty`](src/engine/chunk.cpp#L218-L233)).
- In the vertex shader ([`shaders/chunk.slang`](shaders/chunk.slang#L60-L96)), the GPU uses `SV_VulkanInstanceID` (populated via the indirect command's `firstInstance` field) to index into the `GpuChunkDraw` table, dereferences the raw `uint* vertex_buffer` pointer directly in VRAM, and unpacks the vertex in registers.
- Because every face is a 4-vertex quad, the entire engine binds a **single shared index buffer** (`get_or_create_quad_index_buffer`) containing repeating `0, 1, 2, 2, 3, 0` indices for all chunks.

### GPU Compute Frustum Culling & Single Indirect Draw
Before rendering the scene ([`Engine::render`](src/engine/engine.cpp#L294-L384)):
1. A compute shader ([`shaders/cull.slang`](shaders/cull.slang)) tests every active chunk's $32\times 32\times 32\text{ m}$ bounding box against the 6 camera frustum planes in parallel on the GPU.
2. Using **wave intrinsics** (`WaveActiveCountBits`, `WavePrefixCountBits`, `WaveActiveSum`), each GPU subgroup compacts visible draw commands and performs only **one atomic add per wave** (`WaveIsFirstLane()`) to append `VkDrawIndexedIndirectCommand` entries and accumulate visible triangle stats.
3. The CPU then issues **one** [`draw_indexed_indirect_count`](src/engine/engine.cpp#L375) call for the entire world. The GPU reads both the command list and the visible draw count directly from VRAM without any CPU-GPU round-trip.

### Procedural Face Texturing & Distance LOD Shading
In [`shaders/chunk.slang`](shaders/chunk.slang#L99-L144), block materials are stored in a tiny GPU buffer built at compile time from [`BLOCK_REGISTRY`](src/engine/block.hpp#L173-L234). Within `128 m` of the camera, the fragment shader generates a crisp $16\times 16$ procedural texel checkerboard and subtle block-edge border on each face (including alpha-cutout leaves). Beyond `128 m`, it transitions to a unified face color to prevent distant aliasing and save ALU cycles, combined with directional sun lighting, sky-hemisphere ambient light, Blinn-Phong water highlights, and per-vertex ambient occlusion.

---

## 2. Procedural Chunk Generation with SIMD Noise

World generation lives in [`src/generation/`](src/generation/) and is built on **FastNoise2** with runtime **AVX2 / AVX-512 SIMD** dispatch. All terrain parameters are exposed as named `constexpr` constants in [`src/generation/terrain_config.hpp`](src/generation/terrain_config.hpp).

### 2D Terrain Height Model: Continentality, Erosion, and Peaks
For each $(x, z)$ column, [`TerrainNoiseNodes`](src/generation/noise_nodes.hpp) evaluates three 2D noise graphs in SIMD grids:
1. **Continentality** (`SuperSimplex` Fractal FBm + Domain Warping): shapes large, continuous continents and wide ocean basins in $[0, 1]$. In [`src/generation/heightmap.hpp`](src/generation/heightmap.hpp#L28-L111), a `consteval` **Fritsch-Carlson monotonic cubic Hermite spline** precomputes smooth polynomial segments at compile time to map continentality from deep ocean floors ($y = -100$) up continental cliffs, shallow underwater shelves, beaches crossing sea level ($y = 0$), rolling plains, and high inland plateaus ($y = 150$).
2. **Erosion** (`Simplex` Fractal Ridged + Progressive Domain Warping): produces sharp, irregular valleys and height reductions to break up uniform terrain.
3. **Peaks** (Dual `CellularDistance` Euclidean-squared domes masked into mountain chains + domain warping + dynamic pointy/flat-top peaks and plateau shaping): adds isolated circular peaks and connected mountain ridges with up to $+115$ blocks of elevation bonus.

### 3D Hollow Spaghetti Caves & Chambers
Underground caves (`Lod <= MAX_CAVE_LOD`) are carved by two 3D noise fields built from [`TerrainNoiseNodes::make_cave_3d_node`](src/generation/noise_nodes.hpp#L242-L303):
- To form true 3D tunnels rather than isolated 1-block holes around solid pockets, each cave node intersects two 3D `FractalRidged` sheets (`FastNoise::Min`) modulated by a slow 3D regional bias. In 3D space, the intersection of two ridge sheets forms continuous 1D curves (cylindrical tubes).
- A two-stage clamped remap plateaus the entire tube core at `0.50` while allowing extreme hotspots to rise to `1.0`.
- **Spaghetti caves** carve voxels where $|n_1 - 0.50| < 0.045$, hollowing out smooth 4–6 block wide cylindrical tunnels.
- **Large caves** carve voxels where $n_2 > 0.80$ deep underground, while surface-crust and seabed depth rules in [`is_cave_carved`](src/generation/painter.hpp#L11-L28) prevent ocean leaks and keep surface cave entrances rare.

### Single-Pass Column Painting and Trees
[`ChunkGenerator::generate`](src/generation/chunkgenerator.cpp#L200-L406) generates each chunk in a **single pass per column**:
- **Fast Early Exits & 512×512 Column Cache**: each worker caches 2D column height summaries in a direct-mapped $512\times 512$ cache (`m_column_cache`). Chunks entirely above `max_ceiling_height` (sky air), underwater chunks above the ocean floor (`fill(Water)`), and deep underground chunks at distant LODs (`fill(Stone)`) return immediately without allocating or evaluating 3D noises.
- **3D Cave Noise Skipping**: Shallow seabed chunks skip both 3D cave noises entirely. surface-crust chunks evaluate `cave_large_3d` first and skip `cave_spaghetti_3d` (~95% of surface chunks) whenever no sample reaches the entrance threshold.
- **Surface Painting** ([`src/generation/painter.hpp`](src/generation/painter.hpp#L30-L85)): selects `Clay` and `Sand` on ocean floors, noise-patched `Sand` on beaches, `Grass` with a `Dirt` sub-layer (automatically turning to `Dirt` whenever a solid block or uncarved cave roof sits above), bare `Stone` on steep cliff slopes, and `Snow` above a noise-modulated snowline.
- **Cross-Chunk Trees** ([`src/generation/tree_generator.hpp`](src/generation/tree_generator.hpp)): trees spawn deterministically on flat grass columns using a 2D integer hash over jittered $8\times 8$ cells (aligned to 2-block multiples so LOD 1 and LOD 2 match) after verifying that no cave carved away the root block. Padded border columns (`COLUMN_PADDING`) allow tree leaf canopies to overhang seamlessly across chunk boundaries.

---

## 3. Chunk Meshing & 32-Bit Vertex Packing

Chunk meshing ([`Chunk<Lod>::generate_mesh`](src/engine/chunk.cpp#L387-L445)) turns voxel arrays into compact GPU quad buffers:
- **4-Byte Packed Vertex** ([`pack_vertex`](src/engine/chunk.hpp#L18-L32)):
  - Bits `0..5` (6 bits): local $X \in [0..32]$
  - Bits `6..11` (6 bits): local $Y \in [0..32]$
  - Bits `12..17` (6 bits): local $Z \in [0..32]$
  - Bits `18..20` (3 bits): face normal direction $[0..5]$
  - Bits `21..22` (2 bits): corner Ambient Occlusion level $[0..3]$
  - Bits `23..24` (2 bits): face corner UV index $[0..3]$
  - Bits `25..31` (7 bits): `BlockID` material index $[0..127]$
- **Crack-Free Cross-LOD Boundary Culling**: a chunk's 6 neighbors may be at the same LOD, a coarser LOD, a finer LOD, pure air, or solid ([`NeighborChunk`](src/engine/chunk.hpp#L139-L176) using `std::variant`). [`NeighborChunk::should_render_boundary_face`](src/engine/chunk.cpp#L321-L347) sub-samples finer neighbors across the macro-block face and only culls a boundary quad when the neighbor completely covers it, preventing gaps between LOD rings.
- **Anisotropic Voxel Ambient Occlusion**: [`compute_corner_ao`](src/engine/chunk.cpp#L66-L116) samples the 3 adjacent voxels around each face corner. [`emit_face`](src/engine/chunk.cpp#L119-L147) automatically flips the quad's triangle split diagonal whenever `ao[0] + ao[2] < ao[1] + ao[3]` so lighting gradients interpolate without artifacts.
- **Zero-Allocation Scratch Buffers & VRAM Reuse**: meshing workers write into reusable scratch vectors (`m_mesh_vertex_scratch`) so resting chunks in the pool hold zero `std::vector` heap capacity, and [`ChunkMesh::upload_to_gpu`](src/engine/chunk.cpp#L244-L290) reuses existing GPU vertex buffers in place whenever a remeshed chunk fits within its previous byte capacity.

---

## 4. World Manager, 3D Portal BFS & Custom Containers

The [`WorldManager`](src/engine/worldmanager.hpp) orchestrates streaming, visibility, and memory across a massive sliding 3D window (`1025 × 1025 × 40` chunks = **42 million grid cells**).

### 12-Byte Toroidal Grid Cells
Instead of using a slow `std::unordered_map<glm::ivec3, Chunk>`, the active world is indexed in $O(1)$ via toroidal modulo wrapping (`wrap_xz`) over a flat array of 12-byte [`GridCell`](src/engine/worldmanager.hpp#L176-L204) structs:
- `tag` (4 bits): encodes whether the cell is unloaded (`0`), a real chunk at `LOD 1..5`, or a 100% air chunk evaluated at `LOD 1..5`.
- `reached_faces` (6 bits): tracks which of the 6 chunk faces have been reached by the visibility search.
- `frontier_stamp` (22 bits): epoch counter for $O(1)$ deduplication without clearing visited arrays.
- `index` (`u32`): slot index inside the corresponding `ChunkPool<Lod>`.
- `loaded_epoch` (`u32`): monotonic timestamp used to reconcile concurrent background BFS results.

### 21-Bit Face-to-Face Portal Connectivity & Cave Culling
In a naive voxel engine, underground caves generate millions of unseen triangles below the player's feet. In **voxel-lite**:
1. When a chunk with caves finishes generating, [`Chunk::recalculate_portal_mask`](src/engine/chunk.hpp#L285-L384) runs a fast boundary-seeded 6-connected **3D flood-fill** over non-opaque voxels and encodes a **21-bit mask** ([`src/engine/block.hpp`](src/engine/block.hpp#L62-L128)): 6 bits for open faces + 15 bits for every pair of faces $(A, B)$ that are connected through the interior of the chunk.
2. Why face pairs matter: a surface chunk has open sky air on its `Top` face and might also have a cave touching its `Bottom` face, separated by 20 blocks of solid stone. A simple 6-bit face mask would falsely connect `Top -> Bottom` and leak the sky search into every underground cave. With the 15 face-pair bits, [`reachable_faces_from(portals, entry_face)`](src/engine/block.hpp#L102-L115) knows `Top` only connects to `Left/Right/Front/Back`, keeping the cave on `Bottom` sealed.
3. **Asynchronous Background BFS** ([`WorldManager::execute_background_bfs`](src/engine/worldmanager.cpp#L494-L660)): a low-priority background task snapshots only the active render window and traverses connected face-to-face portals outward from the camera's near sphere (`NEAR_RADIUS = 3`). Any cave system not connected by an open portal path to the camera neighborhood is never generated, or is automatically unloaded when the player moves away.
4. **Sky-Air Column Fast-Fill** ([`WorldManager::fill_air_column_above`](src/engine/worldmanager.cpp#L779-L801)): as soon as a surface chunk reaches open sky above `column_max_height`, the engine instantly marks the entire vertical column of chunks above it up to `MAX_CHUNK_Y` as pure air without running the generator on them.

### Custom `ChunkPool<Lod>` & `LodPoolGroup`
Each LOD level has its own compile-time-sized [`ChunkPool<Lod>`](src/engine/containers.hpp#L57-L164) managed by [`LodPoolGroup`](src/engine/containers.hpp#L166-L229):
- Capacities are computed at compile time by [`compute_lod_pool_capacity<Lod>()`](src/engine/containers.hpp#L32-L55) from the hollow volume between nested 3D LOD boxes (`LOD_BOX_RADII = {24, 48, 96, 128, 512}`).
- Allocation and deallocation are $O(1)$ using a freelist stack and a swap-and-pop active slot array (`m_active_slots` / `m_slot_to_active`), with `Chunk<Lod>` storage lazily allocated on first slot use and reused across unloads.

---

## 5. LOD System, Multithreading (`enkiTS`) & C++23 Architecture

### 5-Level Voxel LODs
Every chunk covers the exact same $32\times 32\times 32\text{ m}$ world volume, but [`Chunk<Lod>`](src/engine/chunk.hpp#L194-L442) templates the resolution on `Lod` ($1..5$):

| LOD | Max Radius (Chunks) | Voxel Scale | Grid Resolution | Voxels per Chunk | Memory per `m_blocks` |
| :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | 24 (768 m) | 1 m | $32^3$ | 32,768 | 32 KB |
| **2** | 48 (1.5 km) | 2 m | $16^3$ | 4,096 | 4 KB |
| **3** | 96 (3.0 km) | 4 m | $8^3$ | 512 | 512 B |
| **4** | 128 (4.1 km) | 8 m | $4^3$ | 64 | 64 B |
| **5** | 512 (16.4 km) | 16 m | $2^3$ | 8 | 8 B |

### Hybrid Frame-Budgeted + Async Worker Scheduling (`enkiTS`)
Chunk generation, meshing, and portal BFS are scheduled across worker threads using Doug Binks' lock-free [**enkiTS**](https://github.com/dougbinks/enkiTS) task scheduler ([`WorldManager::update`](src/engine/worldmanager.cpp#L329-L413)):
- **Immediate Inner-Ring Shell Detection**: when the camera crosses a chunk boundary, [`refresh_queues_for_new_origin`](src/engine/worldmanager.cpp#L164-L273) scans only the 1-chunk-thick entering/exiting faces of the inner LOD boxes (`LOD 1..3`) in around 50 µs and re-sorts the queues by distance to the new camera position so nearby terrain updates immediately without waiting for the full-world BFS.
- **Column-Grouped Worker Execution**: [`start_background_gen`](src/engine/worldmanager.cpp#L868-L949) sorts each batch's worker execution order by `(target_lod, cx, cz, cy)` so vertical chunks in the same $(x, z)$ column run back-to-back on the same thread and share the 2D noise evaluation.
- **Wall-Clock Frame Budget + Background Overlap**: each frame executes compact parallel micro-batches up to a wall-clock budget (`GEN_TIME_BUDGET_MS = 7.5 ms`, `FRAME_WORK_TIME_BUDGET_MS = 12.0 ms`) to maintain 60+ FPS during high-speed flight, then launches an asynchronous generation and mesh batch that runs in the background while Vulkan renders and presents the frame.
- **Hole-Free LOD Transitions**: when a chunk changes LOD, its new slot inherits the previous LOD's GPU mesh handle ([`ChunkMesh::inherit_gpu_resources`](src/engine/chunk.hpp#L65-L73)). If the new LOD opens a portal to an unloaded neighbor, that neighbor is generated urgently and remeshing is deferred ([`has_unloaded_portal_neighbor`](src/engine/worldmanager.cpp#L294-L327)) until the neighbor is loaded, preventing temporary holes on mountain slopes.

### C++23 & Build Optimizations
- **Modern C++23**: uses concepts and `requires` clauses, deducing `this` (explicit object parameters), `std::variant` with the `Overload` pattern, `consteval` spline and material table generation, `std::ranges`, `<bit>` (`std::countr_zero`, `std::popcount`), and `<print>`.
- **Zero-Cost Build Flags** ([`CMakeLists.txt`](CMakeLists.txt)): Compiled with `-fno-exceptions`, `-fno-rtti`, `-ffast-math` equivalents (`-fno-math-errno -fno-trapping-math -ffp-contract=fast`), **AVX2 / AVX-512** (`-march=native` / `x86-64-v4`), and **Link-Time Optimization (ThinLTO / IPO)**, with `mimalloc` overriding global `new`/`delete`.

---

## 6. Building & Running

All dependencies (**SDL3**, **Vulkan-Headers**, **volk**, **VulkanMemoryAllocator**, **vk-bootstrap**, **glm**, **enkiTS**, **ImGui**, **mimalloc**, **FastNoise2**, and the **Slang** shader compiler) are fetched automatically via CMake `FetchContent`.

The program can only be built/run if your computer supports Vulkan (most Windows and Linux computers do). To build, you can use the CMake script: 
```bash
cmake -DCOMP=gcc -P build.cmake # needs GCC 14+
cmake -DCOMP=clang -P build.cmake # needs Clang 18+
cmake -DCOMP=msvc -P build.cmake # needs MSVC VS22 17.10+
```
Note: compiling on Windows with Clang or GCC requires Ninja installed. Your compiler must support C++23.

### Controls
- **Left / Right Click**: capture mouse cursor
- **Escape**: release mouse cursor (press again to quit)
- **WASD or ZQSD (fr)**: move horizontally
- **Space / Left Shift**: fly up / down
- **Left Ctrl**: hold to go faster
- **Debug Overlay (ImGui)**: adjust render distance (`4..512` chunks), toggle GPU frustum culling, toggle VSync, and inspect live CPU/GPU frame times, VRAM, RAM, and per-LOD chunk counts.
