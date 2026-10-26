#include "worldmanager.hpp"

#include <cstring>

namespace vl
{

namespace
{

template <typename Fn>
struct ParallelTaskSet : enki::ITaskSet
{
    Fn fn;

    ParallelTaskSet(u32 count, Fn task_fn) : enki::ITaskSet(count), fn(std::move(task_fn))
    {
        m_Priority = enki::TASK_PRIORITY_HIGH;
    }

    void ExecuteRange(enki::TaskSetPartition range, u32 threadnum) override { fn(range, threadnum); }
};

} // namespace

WorldManager::WorldManager(i32 seed)
    : m_grid(std::make_unique<GridCell[]>(GRID_CELLS)), m_cell_bfs_state(std::make_unique<u16[]>(GRID_CELLS)), m_bfs_task(this),
      m_bfs_snapshot(std::make_unique<u16[]>(GRID_CELLS)), m_bfs_visited_bits(std::make_unique<u64[]>(BITSET_WORDS))
{
    m_scheduler.Initialize();
    const u32 num_threads = std::max(1u, m_scheduler.GetNumTaskThreads());
    m_worker_generators.reserve(num_threads);
    for (u32 i = 0; i < num_threads; ++i) m_worker_generators.emplace_back(seed);

    m_bfs_queue.reserve(131072);
    m_bfs_out_candidates.reserve(65536);
    m_bfs_out_unwanted.reserve(4096);

    m_frontier_during_bfs.reserve(16384);
    m_gen_batch.reserve(MAX_GEN_BATCH_CHUNKS);
    m_mesh_batch.reserve(MAX_MESH_BATCH_CHUNKS);

    // Near-region offsets sorted by distance so the closest chunks are generated first
    for (i32 dy = -NEAR_RADIUS; dy <= NEAR_RADIUS; ++dy)
        for (i32 dz = -NEAR_RADIUS; dz <= NEAR_RADIUS; ++dz)
            for (i32 dx = -NEAR_RADIUS; dx <= NEAR_RADIUS; ++dx) m_near_offsets.emplace_back(dx, dy, dz);

    std::ranges::stable_sort(m_near_offsets, {}, length_squared);
}

WorldManager::~WorldManager()
{
    if (m_bfs_in_flight)
    {
        m_scheduler.WaitforTask(&m_bfs_task);
        m_bfs_in_flight = false;
    }
}

void WorldManager::destroy_all_gpu_resources(vulk::VulkanContext& ctx)
{
    if (m_bfs_in_flight)
    {
        m_scheduler.WaitforTask(&m_bfs_task);
        m_bfs_in_flight = false;
    }
    m_pools.destroy_all_gpu_resources(ctx);
    ChunkDrawTable::instance().reset();
}

bool WorldManager::unload_chunk(const glm::ivec3& coord, vulk::VulkanContext& ctx)
{
    if (!is_in_active_window(coord)) return false;

    const u32 idx = cell_index(coord);
    GridCell& cell = m_grid[idx];
    if (cell.is_real())
    {
        unload_real_chunk(cell.lod(), cell.index, ctx);
        return true;
    }
    if (cell.is_air())
    {
        clear_cell(idx, cell);
        --m_air_chunks_count;
        return true;
    }

    return false;
}

void WorldManager::set_world_block(i32 wx, i32 wy, i32 wz, BlockID id)
{
    constexpr auto P = std::countr_zero<u32>(CHUNK_SIZE);
    const glm::ivec3 chunk_coord {
        wx >> P,
        wy >> P,
        wz >> P,
    };

    if (!is_in_active_window(chunk_coord)) return;

    const u32 idx = cell_index(chunk_coord);
    GridCell& cell = m_grid[idx];

    if (cell.is_air())
    {
        if (id == BlockType::Air) return;

        const u8 target_lod = compute_lod(chunk_coord, m_last_origin);
        const bool pool_full = m_pools.visit_lod(target_lod, [](const auto& p) { return p.full(); });
        if (pool_full) return;

        --m_air_chunks_count;
        m_pools.visit_lod(target_lod, [&](auto& pool) {
            const u32 slot = pool.allocate(chunk_coord);
            auto& chunk = pool.chunk_at(slot);
            chunk.clear();
            chunk.recalculate_portal_mask();
            set_cell_real(idx, cell, target_lod, slot, extract_portal_mask(chunk));
        });
    }

    if (!cell.is_real()) return;

    const u32 lx_m = static_cast<u32>(wx & (CHUNK_SIZE - 1));
    const u32 ly_m = static_cast<u32>(wy & (CHUNK_SIZE - 1));
    const u32 lz_m = static_cast<u32>(wz & (CHUNK_SIZE - 1));

    const bool modified = m_pools.visit_lod(cell.lod(), [&](auto& pool) {
        auto& chunk = pool.chunk_at(cell.index);
        if (chunk.coord != chunk_coord) return false;

        constexpr u32 shift = std::remove_reference_t<decltype(pool)>::LOD - 1u;
        chunk.set_block(lx_m >> shift, ly_m >> shift, lz_m >> shift, id);
        update_cell_portals(idx, cell, extract_portal_mask(chunk));
        return true;
    });

    if (!modified) return;

    mark_dirty(chunk_coord);
    m_search_dirty = true;
    if (m_bfs_in_flight) m_bfs_needs_rerun = true;

    constexpr u32 last_m = CHUNK_SIZE - 1u;
    if (lx_m == 0u) mark_dirty(chunk_coord + to_offset(Direction::Left));
    if (lx_m == last_m) mark_dirty(chunk_coord + to_offset(Direction::Right));
    if (ly_m == 0u) mark_dirty(chunk_coord + to_offset(Direction::Down));
    if (ly_m == last_m) mark_dirty(chunk_coord + to_offset(Direction::Up));
    if (lz_m == 0u) mark_dirty(chunk_coord + to_offset(Direction::Back));
    if (lz_m == last_m) mark_dirty(chunk_coord + to_offset(Direction::Front));
}

void WorldManager::update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx)
{
    const i32 rd = m_render_distance;

    const glm::ivec3 cam_chunk {
        static_cast<i32>(std::floor(cam_world_pos.x / static_cast<f32>(CHUNK_SIZE))),
        static_cast<i32>(std::floor(cam_world_pos.y / static_cast<f32>(CHUNK_SIZE))),
        static_cast<i32>(std::floor(cam_world_pos.z / static_cast<f32>(CHUNK_SIZE))),
    };

    const glm::ivec3 origin {cam_chunk.x, std::clamp(cam_chunk.y, MIN_CHUNK_Y, MAX_CHUNK_Y), cam_chunk.z};

    const bool first_frame = !m_has_last_origin;
    const bool window_changed =
        first_frame || origin.x != m_last_origin.x || origin.z != m_last_origin.z || rd != m_last_render_distance;
    const bool origin_changed = window_changed || origin.y != m_last_origin.y;

    // Out-of-range chunks are unloaded before updating the active window
    if (window_changed && m_has_last_origin) unload_out_of_range(origin, rd, ctx);

    m_last_origin = origin;
    m_last_render_distance = rd;
    m_has_last_origin = true;

    if (origin_changed)
    {
        m_search_dirty = true;
        if (m_bfs_in_flight) m_bfs_needs_rerun = true;
    }

    // Seed the near region immediately on the first frame so generation starts without waiting for the first BFS
    if (first_frame)
    {
        for (const glm::ivec3& off : m_near_offsets)
        {
            const i32 y = origin.y + off.y;
            if (!is_y_in_world(y) || std::abs(off.x) > rd || std::abs(off.z) > rd) continue;
            const glm::ivec3 c {origin.x + off.x, y, origin.z + off.z};
            const u8 lod = compute_lod(c, origin);
            m_gen_queues[lod - 1u].push_back(c);
        }
    }

    // Harvest completed background BFS results without blocking
    if (m_bfs_in_flight && m_bfs_task.GetIsComplete()) harvest_background_bfs(origin, ctx);

    // Launch a new background BFS only when camera origin changed or a portal closed
    if (!m_bfs_in_flight && m_search_dirty)
    {
        const auto now = Clock::now();
        const f64 since_last_ms = std::chrono::duration<f64, std::milli>(now - m_last_bfs_time).count();
        if (origin_changed || first_frame || since_last_ms >= BFS_MIN_INTERVAL_MS) start_background_bfs(origin, rd);
    }

    generate_wanted_chunks(origin, rd, ctx);
    process_mesh_queues(ctx);
}

void WorldManager::start_background_bfs(const glm::ivec3& origin, i32 rd)
{
    std::memcpy(m_bfs_snapshot.get(), m_cell_bfs_state.get(), GRID_CELLS * sizeof(u16));

    m_bfs_origin = origin;
    m_bfs_rd = rd;
    m_bfs_start_epoch = m_current_epoch;
    m_bfs_needs_rerun = false;
    m_search_dirty = false;
    m_bfs_in_flight = true;
    m_last_bfs_time = Clock::now();
    ++m_frontier_stamp;

    m_frontier_during_bfs.clear();
    m_scheduler.AddTaskSetToPipe(&m_bfs_task);
}

void WorldManager::harvest_background_bfs(const glm::ivec3& origin, vulk::VulkanContext& ctx)
{
    m_bfs_in_flight = false;

    // Unload cells that were loaded when the BFS started and were not reached by the portal traversal
    for (const glm::ivec3& coord : m_bfs_out_unwanted)
    {
        if (!is_in_active_window(coord)) continue;

        if (std::abs(coord.x - origin.x) <= NEAR_RADIUS && std::abs(coord.y - origin.y) <= NEAR_RADIUS &&
            std::abs(coord.z - origin.z) <= NEAR_RADIUS)
        {
            continue;
        }

        const u32 idx = cell_index(coord);
        GridCell& cell = m_grid[idx];
        if (cell.is_unloaded() || cell.loaded_epoch > m_bfs_start_epoch) continue;

        if (cell.is_real())
        {
            unload_real_chunk(cell.lod(), cell.index, ctx);
        }
        else if (cell.is_air())
        {
            clear_cell(idx, cell);
            --m_air_chunks_count;
        }
    }

    // Sort any frontier candidates discovered on the main thread while the BFS was running so closest come first
    std::ranges::stable_sort(m_frontier_during_bfs, {}, [&](const glm::ivec3& c) { return distance_squared(c, origin); });

    for (auto& q : m_gen_queues) q.clear();
    const u32 stamp = ++m_frontier_stamp;
    GridCell* const grid = m_grid.get();

    auto enqueue_candidate = [&](const glm::ivec3& c) {
        if (!is_in_active_window(c)) return;
        GridCell& cell = grid[cell_index(c)];
        if (cell.frontier_stamp == stamp) return;

        const u8 target_lod = compute_lod(c, origin);
        if (!cell.is_unloaded() && !cell.needs_lod_update(target_lod)) return;

        cell.frontier_stamp = stamp;
        m_gen_queues[target_lod - 1u].push_back(c);
    };

    for (const glm::ivec3& c : m_frontier_during_bfs) enqueue_candidate(c);
    for (const glm::ivec3& c : m_bfs_out_candidates) enqueue_candidate(c);

    m_frontier_during_bfs.clear();
    m_bfs_out_candidates.clear();
    m_bfs_out_unwanted.clear();

    if (m_bfs_needs_rerun)
    {
        m_bfs_needs_rerun = false;
        m_search_dirty = true;
    }
}

void WorldManager::execute_background_bfs()
{
    const glm::ivec3 origin = m_bfs_origin;
    const i32 rd = m_bfs_rd;
    const u16* const snap = m_bfs_snapshot.get();

    BitSpan visited {std::span<u64> {m_bfs_visited_bits.get(), BITSET_WORDS}};
    visited.clear();

    m_bfs_queue.clear();
    m_bfs_out_candidates.clear();
    m_bfs_out_unwanted.clear();

    std::array<i32, GRID_XZ> wrap_x {};
    std::array<i32, GRID_XZ> wrap_z {};
    for (i32 i = 0; i < GRID_XZ; ++i)
    {
        wrap_x[static_cast<usize>(i)] = wrap_xz(origin.x + i - MAX_RENDER_DIST);
        wrap_z[static_cast<usize>(i)] = wrap_xz(origin.z + i - MAX_RENDER_DIST);
    }

    auto snap_idx = [&](i32 dx, i32 y, i32 dz) -> u32 {
        const i32 wz = wrap_z[static_cast<usize>(dz + MAX_RENDER_DIST)];
        const i32 wx = wrap_x[static_cast<usize>(dx + MAX_RENDER_DIST)];
        return static_cast<u32>((((y - MIN_CHUNK_Y) * GRID_XZ + wz) * GRID_XZ) + wx);
    };

    // Every chunk in the near region is wanted regardless of portals
    for (const glm::ivec3& off : m_near_offsets)
    {
        const i32 y = origin.y + off.y;
        if (!is_y_in_world(y) || std::abs(off.x) > rd || std::abs(off.z) > rd) continue;

        const u32 idx = snap_idx(off.x, y, off.z);
        if (visited.test_and_set(idx)) m_bfs_queue.push_back(SearchNode {.dx = off.x, .dz = off.z, .y = y});
    }

    for (usize head = 0; head < m_bfs_queue.size(); ++head)
    {
        const SearchNode node = m_bfs_queue[head];
        const i32 dx = node.dx;
        const i32 dz = node.dz;
        const i32 y = node.y;

        const u16 state = snap[snap_idx(dx, y, dz)];
        const u32 tag = state & 0xFu;
        const glm::ivec3 coord {origin.x + dx, y, origin.z + dz};

        if (tag == GridCell::TAG_UNLOADED)
        {
            m_bfs_out_candidates.push_back(coord);
            continue;
        }

        const u8 target_lod = compute_lod(coord, origin);
        if (tag <= MAX_LOD)
        {
            if (tag != target_lod) m_bfs_out_candidates.push_back(coord);
        }
        else
        {
            const u8 air_lod = static_cast<u8>(tag - MAX_LOD);
            if (target_lod < air_lod) m_bfs_out_candidates.push_back(coord);
        }

        const u32 mask = (state >> 4u) & GridCell::ALL_PORTALS;
        if (mask == 0u) continue;

        for (Direction d : ALL_DIRECTIONS)
        {
            if ((mask & direction_bit(d)) == 0u) continue;

            const glm::ivec3 o = to_offset(d);
            const i32 ndx = dx + o.x;
            const i32 ny = y + o.y;
            const i32 ndz = dz + o.z;

            if (!is_y_in_world(ny) || std::abs(ndx) > rd || std::abs(ndz) > rd) continue;

            const u32 nidx = snap_idx(ndx, ny, ndz);
            if (visited.test_and_set(nidx)) m_bfs_queue.push_back(SearchNode {.dx = ndx, .dz = ndz, .y = ny});
        }
    }

    // Identify any loaded cells in the active window that were not reached by the BFS
    for (i32 y = MIN_CHUNK_Y; y <= MAX_CHUNK_Y; ++y)
    {
        const i32 y_base = (y - MIN_CHUNK_Y) * GRID_XZ;
        for (i32 dz = -rd; dz <= rd; ++dz)
        {
            const i32 row_base = (y_base + wrap_z[static_cast<usize>(dz + MAX_RENDER_DIST)]) * GRID_XZ;
            for (i32 dx = -rd; dx <= rd; ++dx)
            {
                const u32 idx = static_cast<u32>(row_base + wrap_x[static_cast<usize>(dx + MAX_RENDER_DIST)]);
                const u16 state = snap[idx];
                if ((state & 0xFu) == GridCell::TAG_UNLOADED) continue;

                if (!visited.test(idx)) m_bfs_out_unwanted.emplace_back(origin.x + dx, y, origin.z + dz);
            }
        }
    }

    // Sort candidates closest first so near terrain and LOD transitions always complete before distant horizon chunks
    std::ranges::stable_sort(m_bfs_out_candidates, {}, [&](const glm::ivec3& c) { return distance_squared(c, origin); });
}

bool WorldManager::expand_frontier_from(
    const glm::ivec3& coord, const GridCell& start_cell, const glm::ivec3& origin, i32 rd, bool high_priority
)
{
    const u32 mask = start_cell.portal_mask;
    if (mask == 0u) return false;

    const u32 stamp = m_frontier_stamp;
    GridCell* const grid = m_grid.get();
    bool revealed_unloaded = false;

    for (Direction d : ALL_DIRECTIONS)
    {
        if ((mask & direction_bit(d)) == 0u) continue;

        const glm::ivec3 ncoord = coord + to_offset(d);
        if (!is_y_in_world(ncoord.y) || std::abs(ncoord.x - origin.x) > rd || std::abs(ncoord.z - origin.z) > rd) continue;

        GridCell& ncell = grid[cell_index(ncoord)];
        if (!high_priority && ncell.frontier_stamp == stamp) continue;

        const u8 target_lod = compute_lod(ncoord, origin);
        if (ncell.is_unloaded() || ncell.needs_lod_update(target_lod))
        {
            if (ncell.is_unloaded()) revealed_unloaded = true;
            ncell.frontier_stamp = stamp;

            auto& q = m_gen_queues[target_lod - 1u];
            if (high_priority)
            {
                m_urgent_gen_queue.push_back(ncoord);
                q.push_front(ncoord);
            }
            else
            {
                q.push_back(ncoord);
            }

            if (m_bfs_in_flight) m_frontier_during_bfs.push_back(ncoord);
        }
    }

    return revealed_unloaded;
}

void WorldManager::fill_air_column_above(i32 cx, i32 cz, i32 col_max_height, const glm::ivec3& origin, i32 rd)
{
    const i32 min_air_cy = std::max(MIN_CHUNK_Y, (col_max_height >> 5) + 1);
    for (i32 y = MAX_CHUNK_Y; y >= min_air_cy; --y)
    {
        const glm::ivec3 c {cx, y, cz};
        const u32 idx = cell_index(c);
        GridCell& cell = m_grid[idx];

        if (!cell.is_unloaded()) continue;

        const u8 target_lod = compute_lod(c, origin);
        const bool guaranteed_all_lods =
            (target_lod == 1u) || ((y * CHUNK_SIZE) > (col_max_height + ChunkGenerator::SURFACE_AIR_MARGIN));
        const u8 air_lod = guaranteed_all_lods ? 1u : target_lod;

        set_cell_air(idx, cell, air_lod);
        ++m_air_chunks_count;
        expand_frontier_from(c, cell, origin, rd);
        on_chunk_loaded(c);
    }
}

void WorldManager::unload_real_chunk(u8 lod, u32 slot, vulk::VulkanContext& ctx)
{
    m_pools.visit_lod(lod, [&](auto& pool) {
        const glm::ivec3 coord = pool.chunk_at(slot).coord;
        const u32 idx = cell_index(coord);
        clear_cell(idx, m_grid[idx]);
        pool.free(slot, ctx);
    });
}

void WorldManager::unload_cell_at(const glm::ivec3& coord, vulk::VulkanContext& ctx)
{
    const u32 idx = cell_index(coord);
    GridCell& cell = m_grid[idx];
    if (cell.is_unloaded()) return;

    if (cell.is_real())
    {
        unload_real_chunk(cell.lod(), cell.index, ctx);
    }
    else if (cell.is_air())
    {
        clear_cell(idx, cell);
        --m_air_chunks_count;
    }
}

void WorldManager::unload_out_of_range(const glm::ivec3& new_origin, i32 new_rd, vulk::VulkanContext& ctx)
{
    const i32 old_min_x = m_last_origin.x - m_last_render_distance;
    const i32 old_max_x = m_last_origin.x + m_last_render_distance;
    const i32 old_min_z = m_last_origin.z - m_last_render_distance;
    const i32 old_max_z = m_last_origin.z + m_last_render_distance;

    const i32 new_min_x = new_origin.x - new_rd;
    const i32 new_max_x = new_origin.x + new_rd;
    const i32 new_min_z = new_origin.z - new_rd;
    const i32 new_max_z = new_origin.z + new_rd;

    for (i32 cz = old_min_z; cz <= old_max_z; ++cz)
    {
        if (cz < new_min_z || cz > new_max_z)
        {
            for (i32 cy = MIN_CHUNK_Y; cy <= MAX_CHUNK_Y; ++cy)
                for (i32 cx = old_min_x; cx <= old_max_x; ++cx) unload_cell_at(glm::ivec3 {cx, cy, cz}, ctx);
        }
        else
        {
            const i32 left_end = std::min(old_max_x, new_min_x - 1);
            const i32 right_start = std::max(old_min_x, new_max_x + 1);
            if (old_min_x <= left_end || right_start <= old_max_x)
            {
                for (i32 cy = MIN_CHUNK_Y; cy <= MAX_CHUNK_Y; ++cy)
                {
                    for (i32 cx = old_min_x; cx <= left_end; ++cx) unload_cell_at(glm::ivec3 {cx, cy, cz}, ctx);
                    for (i32 cx = right_start; cx <= old_max_x; ++cx) unload_cell_at(glm::ivec3 {cx, cy, cz}, ctx);
                }
            }
        }
    }
}

void WorldManager::generate_wanted_chunks(const glm::ivec3& origin, i32 rd, vulk::VulkanContext& ctx)
{
    // Pass 0 drains normal per-LOD queues (closest LODs first)
    // Pass 1 immediately generates any urgent neighbor chunks revealed by LOD transitions in pass 0 on the same frame
    for (u32 pass = 0; pass < 2u; ++pass)
    {
        const bool urgent_only = (pass > 0u);
        if (urgent_only && m_urgent_gen_queue.empty()) break;

        u32 batch_work = 0;
        m_gen_batch.clear();

        while (batch_work < GEN_WORK_BUDGET_PER_FRAME && m_gen_batch.size() < MAX_GEN_BATCH_CHUNKS)
        {
            glm::ivec3 coord {};
            bool from_urgent = false;
            bool found = false;

            if (!m_urgent_gen_queue.empty())
            {
                coord = m_urgent_gen_queue.front();
                m_urgent_gen_queue.pop_front();
                from_urgent = true;
                found = true;
            }
            else if (!urgent_only)
            {
                for (u8 lod = 1; lod <= MAX_LOD; ++lod)
                {
                    auto& q = m_gen_queues[lod - 1u];
                    if (!q.empty())
                    {
                        coord = q.front();
                        q.pop_front();
                        found = true;
                        break;
                    }
                }
            }

            if (!found) break;
            if (!is_in_active_window(coord)) continue;

            const u8 target_lod = compute_lod(coord, origin);
            const u32 idx = cell_index(coord);
            GridCell& cell = m_grid[idx];

            if (!cell.is_unloaded() && !cell.needs_lod_update(target_lod)) continue;

            const bool target_full = m_pools.visit_lod(target_lod, [](const auto& pool) { return pool.full(); });
            if (target_full) continue;

            const bool was_unloaded = cell.is_unloaded();
            const bool was_real = cell.is_real();
            const u8 old_portal_mask = static_cast<u8>(cell.portal_mask);
            bool had_old_mesh = false;

            if (cell.is_air()) --m_air_chunks_count;

            // Allocate the new slot and transfer any existing GPU mesh from the previous LOD so no hole appears
            const u32 new_slot = m_pools.visit_lod(target_lod, [&](auto& new_pool) {
                const u32 s = new_pool.allocate(coord);
                if (was_real)
                {
                    m_pools.visit_lod(cell.lod(), [&](auto& old_pool) {
                        ChunkMesh& old_mesh = old_pool.mesh_at(cell.index);
                        had_old_mesh = old_mesh.meshed_once;
                        new_pool.mesh_at(s).inherit_gpu_resources(old_mesh);
                        old_pool.free(cell.index, ctx);
                    });
                }
                return s;
            });

            // Keep the existing portal mask while the batch runs so neighbor traversals are never blocked
            set_cell_real(idx, cell, target_lod, new_slot, old_portal_mask);
            batch_work += lod_work_cost(target_lod);

            m_gen_batch.push_back(
                GenTaskItem {
                    .coord = coord,
                    .slot = new_slot,
                    .column_max_height = 0,
                    .target_lod = target_lod,
                    .old_portal_mask = old_portal_mask,
                    .was_unloaded = was_unloaded,
                    .was_real = was_real,
                    .had_old_mesh = had_old_mesh,
                    .high_priority = from_urgent || !was_unloaded,
                    .non_air = false,
                    .guaranteed_all_lods = false,
                    .portal_mask = 0u,
                }
            );
        }

        if (m_gen_batch.empty()) break;

        // Generate all terrain chunks in the batch in parallel across worker threads
        ParallelTaskSet gen_task(static_cast<u32>(m_gen_batch.size()), [this](enki::TaskSetPartition range, u32 threadnum) {
            ChunkGenerator& gen = m_worker_generators[threadnum % m_worker_generators.size()];
            for (u32 i = range.start; i < range.end; ++i)
            {
                GenTaskItem& item = m_gen_batch[i];
                m_pools.visit_lod(item.target_lod, [&](auto& pool) {
                    auto& chunk = pool.chunk_at(item.slot);
                    item.non_air = gen.generate(chunk, item.guaranteed_all_lods, item.column_max_height);
                    if (item.non_air) item.portal_mask = extract_portal_mask(chunk);
                });
            }
        });

        m_scheduler.AddTaskSetToPipe(&gen_task);
        m_scheduler.WaitforTask(&gen_task, enki::TASK_PRIORITY_HIGH);

        // Pass 0 (commit all cell states and portal masks before any frontier expansion runs)
        for (GenTaskItem& item : m_gen_batch)
        {
            const u32 idx = cell_index(item.coord);
            GridCell& cell = m_grid[idx];
            if (!item.non_air)
            {
                unload_real_chunk(item.target_lod, item.slot, ctx);
                const u8 air_lod = item.guaranteed_all_lods ? 1u : item.target_lod;
                set_cell_air(idx, cell, air_lod);
                ++m_air_chunks_count;
                item.portal_mask = static_cast<u8>(GridCell::ALL_PORTALS);
            }
            else
            {
                update_cell_portals(idx, cell, item.portal_mask);
                if (item.high_priority)
                    m_remesh_queues[item.target_lod - 1u].push_front(item.coord);
                else if (item.had_old_mesh)
                    m_remesh_queues[item.target_lod - 1u].push_back(item.coord);
                else
                    m_mesh_queues[item.target_lod - 1u].push_back(item.coord);
            }
        }

        // Pass 1 (expand frontier for newly opened portals, fill air columns, and notify neighbors to remesh)
        for (const GenTaskItem& item : m_gen_batch)
        {
            const GridCell& cell = m_grid[cell_index(item.coord)];
            const u8 newly_closed_portals = static_cast<u8>(item.old_portal_mask & ~item.portal_mask);

            const bool reaches_sky = !item.non_air || (((item.portal_mask & direction_bit(Direction::Up)) != 0u) &&
                                                       (((item.coord.y + 1) * CHUNK_SIZE) > item.column_max_height));
            if (item.was_unloaded && reaches_sky)
                fill_air_column_above(item.coord.x, item.coord.z, item.column_max_height, origin, rd);

            // Chunks changing LOD push any newly revealed neighbors to the front of the queue
            const bool expand_urgent = !item.was_unloaded && (pass == 0u);
            expand_frontier_from(item.coord, cell, origin, rd, expand_urgent);

            if (newly_closed_portals != 0u)
            {
                m_search_dirty = true;
                if (m_bfs_in_flight) m_bfs_needs_rerun = true;
            }

            on_chunk_loaded(item.coord, item.high_priority);
        }
    }
}

void WorldManager::on_chunk_loaded(const glm::ivec3& coord, bool high_priority)
{
    for (const glm::ivec3& o : DIRECTION_OFFSETS)
    {
        const glm::ivec3 n = coord + o;
        const GridCell* cell = find_real_cell(n);
        if (cell == nullptr) continue;

        const u8 n_lod = cell->lod();
        m_pools.visit_lod(n_lod, [&](auto& pool) {
            if (!pool.mesh_at(cell->index).meshed_once) return;

            auto& chunk = pool.chunk_at(cell->index);
            if (chunk.mesh_dirty) return;

            chunk.mesh_dirty = true;
            auto& rq = m_remesh_queues[n_lod - 1u];
            if (high_priority)
                rq.push_front(n);
            else
                rq.push_back(n);
        });
    }
}

void WorldManager::mark_dirty(const glm::ivec3& coord)
{
    const GridCell* cell = find_real_cell(coord);
    if (cell == nullptr) return;

    const u8 lod = cell->lod();
    m_pools.visit_lod(lod, [&](auto& pool) {
        auto& chunk = pool.chunk_at(cell->index);
        if (chunk.mesh_dirty) return;

        chunk.mesh_dirty = true;
        m_remesh_queues[lod - 1u].push_front(coord);
    });
}

void WorldManager::process_mesh_queues(vulk::VulkanContext& ctx)
{
    m_mesh_batch.clear();
    u32 mesh_work = 0;

    while (mesh_work < MESH_WORK_BUDGET_PER_FRAME && m_mesh_batch.size() < MAX_MESH_BATCH_CHUNKS)
    {
        glm::ivec3 coord {};
        bool found = false;

        // Always prioritize closer LODs
        for (u8 lod = 1; lod <= MAX_LOD; ++lod)
        {
            auto& remesh_q = m_remesh_queues[lod - 1u];
            if (!remesh_q.empty())
            {
                coord = remesh_q.front();
                remesh_q.pop_front();
                found = true;
                break;
            }

            auto& mesh_q = m_mesh_queues[lod - 1u];
            if (!mesh_q.empty())
            {
                coord = mesh_q.front();
                mesh_q.pop_front();
                found = true;
                break;
            }
        }

        if (!found) break;

        const GridCell* cell = find_real_cell(coord);
        if (cell == nullptr) continue;

        const bool queued = m_pools.visit_lod(cell->lod(), [&](auto& pool) {
            auto& chunk = pool.chunk_at(cell->index);
            if (!chunk.mesh_dirty) return false;
            chunk.mesh_dirty = false;
            return true;
        });

        if (!queued) continue;

        mesh_work += lod_work_cost(cell->lod());

        m_mesh_batch.push_back(
            MeshTaskItem {
                .coord = coord,
                .slot = cell->index,
                .lod = cell->lod(),
                .neighbors = get_mesh_neighbors(coord),
            }
        );
    }

    if (m_mesh_batch.empty()) return;

    // Build all chunk meshes in parallel on CPU worker threads
    ParallelTaskSet mesh_task(static_cast<u32>(m_mesh_batch.size()), [this](enki::TaskSetPartition range, u32) {
        for (u32 i = range.start; i < range.end; ++i)
        {
            const MeshTaskItem& item = m_mesh_batch[i];
            m_pools.visit_lod(item.lod, [&](auto& pool) {
                const auto& chunk = pool.chunk_at(item.slot);
                ChunkMesh& mesh = pool.mesh_at(item.slot);
                chunk.generate_mesh(mesh, item.neighbors);
            });
        }
    });

    m_scheduler.AddTaskSetToPipe(&mesh_task);
    m_scheduler.WaitforTask(&mesh_task, enki::TASK_PRIORITY_HIGH);

    // Upload completed vertex buffers to the GPU on the main thread within the staging buffer budget
    u64 uploaded_bytes = 0;
    for (const MeshTaskItem& item : m_mesh_batch)
    {
        m_pools.visit_lod(item.lod, [&](auto& pool) {
            auto& chunk = pool.chunk_at(item.slot);
            ChunkMesh& mesh = pool.mesh_at(item.slot);
            const u64 mesh_bytes = mesh.vertices.size() * sizeof(PackedVertex);

            if (uploaded_bytes > 0u && uploaded_bytes + mesh_bytes > MAX_UPLOAD_BYTES_PER_FRAME)
            {
                mesh.vertices.clear();
                chunk.mesh_dirty = true;
                m_remesh_queues[item.lod - 1u].push_front(item.coord);
                return;
            }

            mesh.upload_to_gpu(ctx);
            uploaded_bytes += mesh_bytes;
            mesh.vertices.clear();
            mesh.meshed_once = true;
        });
    }
}

} // namespace vl