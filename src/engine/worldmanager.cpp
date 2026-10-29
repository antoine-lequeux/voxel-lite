#include "worldmanager.hpp"

#include <algorithm>

namespace vl
{

WorldManager::WorldManager(i32 seed)
    : m_grid(std::make_unique<GridCell[]>(GRID_CELLS)), m_cell_bfs_state(std::make_unique<u32[]>(GRID_CELLS)), m_bfs_task(this),
      m_bfs_snapshot(std::make_unique<u32[]>(GRID_CELLS)), m_bfs_visited_bits(std::make_unique<u64[]>(BITSET_WORDS)),
      m_gen_task(this), m_mesh_task(this)
{
    enki::TaskSchedulerConfig scheduler_cfg {};
    scheduler_cfg.numTaskThreadsToCreate = std::min(MAX_WORKER_THREAD_COUNT, std::max(1u, enki::GetNumHardwareThreads() - 1u));
    m_scheduler.Initialize(scheduler_cfg);

    const u32 num_threads = std::max(1u, m_scheduler.GetNumTaskThreads());
    m_worker_generators.reserve(num_threads);
    for (u32 i = 0; i < num_threads; ++i) m_worker_generators.emplace_back(seed);

    m_bfs_queue.reserve(131072);
    m_bfs_out_candidates.reserve(65536);
    m_bfs_out_unwanted.reserve(4096);

    m_frontier_during_bfs.reserve(16384);
    m_expand_stack.reserve(1024);
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
    wait_for_all_tasks();
}

void WorldManager::wait_for_all_tasks()
{
    if (m_mesh_in_flight)
    {
        m_scheduler.WaitforTask(&m_mesh_task);
        m_mesh_in_flight = false;
    }
    if (m_gen_in_flight)
    {
        m_scheduler.WaitforTask(&m_gen_task);
        m_gen_in_flight = false;
    }
    if (m_bfs_in_flight)
    {
        m_scheduler.WaitforTask(&m_bfs_task);
        m_bfs_in_flight = false;
    }
}

void WorldManager::destroy_all_gpu_resources(vulk::VulkanContext& ctx)
{
    wait_for_all_tasks();
    m_pools.destroy_all_gpu_resources(ctx);
    ChunkDrawTable::instance().reset();
}

bool WorldManager::unload_chunk(const glm::ivec3& coord, vulk::VulkanContext& ctx)
{
    if (m_mesh_in_flight)
    {
        m_scheduler.WaitforTask(&m_mesh_task);
        harvest_background_mesh(ctx);
    }

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
    if (m_mesh_in_flight)
    {
        m_scheduler.WaitforTask(&m_mesh_task);
        m_mesh_in_flight = false;
    }

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
    // Harvest completed background meshes first
    if (m_mesh_in_flight)
    {
        if (!m_mesh_task.GetIsComplete()) return;
        harvest_background_mesh(ctx);
    }

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

        if (is_y_in_world(origin.y)) m_grid[cell_index(origin)].reached_faces = ALL_OPEN_FACES_MASK;

        // Prioritize the immediate neighborhood around the new camera origin at the front of the LOD queues
        const u32 stamp = next_frontier_stamp();
        for (usize i = m_near_offsets.size(); i > 0u; --i)
        {
            const glm::ivec3& off = m_near_offsets[i - 1u];
            const i32 y = origin.y + off.y;
            if (!is_y_in_world(y) || std::abs(off.x) > rd || std::abs(off.z) > rd) continue;
            const glm::ivec3 c {origin.x + off.x, y, origin.z + off.z};
            GridCell& cell = m_grid[cell_index(c)];
            const u8 lod = compute_lod(c, origin);
            if (cell.is_unloaded() || cell.needs_lod_update(lod))
            {
                cell.frontier_stamp = stamp;
                m_gen_queues[lod - 1u].push_front(c);
            }
        }
    }

    // Harvest completed background BFS results without blocking
    if (m_bfs_in_flight && m_bfs_task.GetIsComplete()) harvest_background_bfs(origin, ctx);

    // Harvest completed background chunk generation batch without blocking
    if (m_gen_in_flight && m_gen_task.GetIsComplete()) harvest_background_gen(origin, rd, ctx);

    // Launch a new background BFS when camera origin changed or portal topology changed
    if (!m_bfs_in_flight && m_search_dirty)
    {
        const auto now = Clock::now();
        const f64 since_last_ms = std::chrono::duration<f64, std::milli>(now - m_last_bfs_time).count();
        if (origin_changed || first_frame || since_last_ms >= BFS_MIN_INTERVAL_MS) start_background_bfs(origin, rd);
    }

    // Launch next background chunk generation and meshing batches on worker threads without waiting
    if (!m_gen_in_flight) start_background_gen(origin);
    if (!m_mesh_in_flight) start_background_mesh();
}

void WorldManager::start_background_bfs(const glm::ivec3& origin, i32 rd)
{
    const i32 win_xz = (2 * rd) + 1;
    std::array<i32, GRID_XZ> wrap_x {};
    std::array<i32, GRID_XZ> wrap_z {};
    for (i32 i = 0; i < win_xz; ++i)
    {
        wrap_x[static_cast<usize>(i)] = wrap_xz(origin.x + i - rd);
        wrap_z[static_cast<usize>(i)] = wrap_xz(origin.z + i - rd);
    }

    const u32* const src = m_cell_bfs_state.get();
    u32* const dst = m_bfs_snapshot.get();

    // Copy the active render window
    u32 dst_idx = 0u;
    for (i32 y = 0; y < WORLD_HEIGHT_CHUNKS; ++y)
    {
        const i32 y_base = y * GRID_XZ;
        for (i32 iz = 0; iz < win_xz; ++iz)
        {
            const i32 row_base = (y_base + wrap_z[static_cast<usize>(iz)]) * GRID_XZ;
            for (i32 ix = 0; ix < win_xz; ++ix)
                dst[dst_idx++] = src[static_cast<usize>(row_base + wrap_x[static_cast<usize>(ix)])];
        }
    }

    m_bfs_origin = origin;
    m_bfs_rd = rd;
    m_bfs_start_epoch = m_current_epoch;
    m_bfs_needs_rerun = false;
    m_search_dirty = false;
    m_bfs_in_flight = true;
    m_last_bfs_time = Clock::now();
    next_frontier_stamp();

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

    GridCell* const grid = m_grid.get();
    const u32* const snap = m_bfs_snapshot.get();
    const i32 bfs_rd = m_bfs_rd;
    const i32 win_xz = (2 * bfs_rd) + 1;

    auto snap_idx = [&](i32 dx, i32 y, i32 dz) -> u32 {
        return static_cast<u32>((((y - MIN_CHUNK_Y) * win_xz + (dz + bfs_rd)) * win_xz) + (dx + bfs_rd));
    };

    auto sync_reached = [&](const glm::ivec3& c) {
        if (!is_in_active_window(c)) return;
        const i32 dx = c.x - m_bfs_origin.x;
        const i32 dz = c.z - m_bfs_origin.z;
        if (std::abs(dx) > bfs_rd || std::abs(dz) > bfs_rd) return;

        const u8 snap_reached = static_cast<u8>((snap[snap_idx(dx, c.y, dz)] >> 25u) & 0x3Fu);
        const u32 idx = cell_index(c);
        if (grid[idx].loaded_epoch <= m_bfs_start_epoch && origin == m_bfs_origin)
            grid[idx].reached_faces = snap_reached;
        else
            grid[idx].reached_faces |= snap_reached;
    };

    for (const glm::ivec3& off : m_near_offsets)
    {
        const i32 y = m_bfs_origin.y + off.y;
        if (!is_y_in_world(y)) continue;
        sync_reached(glm::ivec3 {m_bfs_origin.x + off.x, y, m_bfs_origin.z + off.z});
    }

    for (const SearchNode& node : m_bfs_queue)
        sync_reached(glm::ivec3 {m_bfs_origin.x + node.dx, node.y, m_bfs_origin.z + node.dz});

    for (const glm::ivec3& c : m_bfs_out_candidates) sync_reached(c);

    // Sort any frontier candidates discovered on the main thread while the BFS was running so closest come first
    std::ranges::stable_sort(m_frontier_during_bfs, {}, [&](const glm::ivec3& c) { return distance_squared(c, origin); });
    if (origin != m_bfs_origin)
        std::ranges::stable_sort(m_bfs_out_candidates, {}, [&](const glm::ivec3& c) { return distance_squared(c, origin); });

    for (auto& q : m_gen_queues) q.clear();
    const u32 stamp = next_frontier_stamp();

    auto enqueue_candidate = [&](const glm::ivec3& c) {
        if (!is_in_active_window(c)) return;
        GridCell& cell = grid[cell_index(c)];
        if (cell.frontier_stamp == stamp) return;

        const u8 target_lod = compute_lod(c, origin);
        if (!cell.is_unloaded() && !cell.needs_lod_update(target_lod)) return;

        cell.frontier_stamp = stamp;
        m_gen_queues[target_lod - 1u].push_back(c);
    };

    // Always seed the current near-origin region first even if the camera moved while the BFS was in flight
    for (const glm::ivec3& off : m_near_offsets)
    {
        const i32 y = origin.y + off.y;
        if (!is_y_in_world(y)) continue;
        enqueue_candidate(glm::ivec3 {origin.x + off.x, y, origin.z + off.z});
    }

    for (const glm::ivec3& c : m_frontier_during_bfs) enqueue_candidate(c);
    for (const glm::ivec3& c : m_bfs_out_candidates) enqueue_candidate(c);

    m_frontier_during_bfs.clear();
    m_bfs_out_candidates.clear();
    m_bfs_out_unwanted.clear();

    // If any chunks were generated while the BFS was running, or a rerun was requested, schedule a follow-up BFS
    if (m_bfs_needs_rerun || m_current_epoch > m_bfs_start_epoch || origin != m_bfs_origin)
    {
        m_bfs_needs_rerun = false;
        m_search_dirty = true;
    }
}

void WorldManager::execute_background_bfs()
{
    const glm::ivec3 origin = m_bfs_origin;
    const i32 rd = m_bfs_rd;
    const i32 win_xz = (2 * rd) + 1;
    const u32 active_cells = static_cast<u32>(win_xz * win_xz * WORLD_HEIGHT_CHUNKS);
    const usize active_words = (active_cells + 63u) / 64u;
    u32* const snap = m_bfs_snapshot.get();

    BitSpan visited {std::span<u64> {m_bfs_visited_bits.get(), active_words}};
    visited.clear();

    m_bfs_queue.clear();
    m_bfs_out_candidates.clear();
    m_bfs_out_unwanted.clear();

    auto snap_idx = [&](i32 dx, i32 y, i32 dz) -> u32 {
        return static_cast<u32>((((y - MIN_CHUNK_Y) * win_xz + (dz + rd)) * win_xz) + (dx + rd));
    };

    auto check_candidate = [&](i32 dx, i32 y, i32 dz, u32 state) {
        const u32 tag = state & 0xFu;
        const glm::ivec3 coord {origin.x + dx, y, origin.z + dz};

        if (tag == GridCell::TAG_UNLOADED)
        {
            m_bfs_out_candidates.push_back(coord);
            return;
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
    };

    // Every chunk in the near region is wanted because it is close enough to the camera
    for (const glm::ivec3& off : m_near_offsets)
    {
        const i32 y = origin.y + off.y;
        if (!is_y_in_world(y) || std::abs(off.x) > rd || std::abs(off.z) > rd) continue;

        const u32 idx = snap_idx(off.x, y, off.z);
        if (visited.test_and_set(idx)) check_candidate(off.x, y, off.z, snap[idx]);
    }

    // Portal extension search starts at the camera chunk and traverses only connected face-to-face portals
    if (is_y_in_world(origin.y))
    {
        const u32 origin_idx = snap_idx(0, origin.y, 0);
        const u32 origin_state = snap[origin_idx];
        snap[origin_idx] = origin_state | (static_cast<u32>(ALL_OPEN_FACES_MASK) << 25u);

        const u32 origin_tag = origin_state & 0xFu;
        if (origin_tag != GridCell::TAG_UNLOADED)
        {
            const u32 origin_portals = (origin_state >> 4u) & GridCell::ALL_PORTALS;
            const u8 origin_out = open_faces_mask(origin_portals);
            if (origin_out != 0u)
            {
                m_bfs_queue.push_back(
                    SearchNode {
                        .dx = 0,
                        .dz = 0,
                        .y = static_cast<i16>(origin.y),
                        .out_dirs = origin_out,
                    }
                );
            }
        }
    }

    for (usize head = 0; head < m_bfs_queue.size(); ++head)
    {
        const SearchNode node = m_bfs_queue[head];
        const i32 dx = node.dx;
        const i32 dz = node.dz;
        const i32 y = node.y;
        const u8 out_dirs = node.out_dirs;

        for (Direction d : ALL_DIRECTIONS)
        {
            if ((out_dirs & direction_bit(d)) == 0u) continue;

            const glm::ivec3 o = to_offset(d);
            const i32 ndx = dx + o.x;
            const i32 ny = y + o.y;
            const i32 ndz = dz + o.z;

            if (!is_y_in_world(ny) || std::abs(ndx) > rd || std::abs(ndz) > rd) continue;

            const u32 nidx = snap_idx(ndx, ny, ndz);
            const u32 n_state = snap[nidx];

            if (visited.test_and_set(nidx)) check_candidate(ndx, ny, ndz, n_state);

            const Direction in_face = opposite(d);
            const u8 in_bit = direction_bit(in_face);
            const u8 prev_reached = static_cast<u8>((n_state >> 25u) & 0x3Fu);
            if ((prev_reached & in_bit) != 0u) continue;

            snap[nidx] = n_state | (static_cast<u32>(in_bit) << 25u);

            const u32 n_tag = n_state & 0xFu;
            if (n_tag == GridCell::TAG_UNLOADED) continue;

            const u32 n_portals = (n_state >> 4u) & GridCell::ALL_PORTALS;
            const u8 n_out = reachable_faces_from(n_portals, in_face);
            if (n_out == 0u) continue;

            u8 prev_out = 0u;
            for (Direction prev_in : ALL_DIRECTIONS)
                if ((prev_reached & direction_bit(prev_in)) != 0u) prev_out |= reachable_faces_from(n_portals, prev_in);

            const u8 new_out = static_cast<u8>(n_out & ~prev_out);
            if (new_out != 0u)
            {
                m_bfs_queue.push_back(
                    SearchNode {
                        .dx = static_cast<i16>(ndx),
                        .dz = static_cast<i16>(ndz),
                        .y = static_cast<i16>(ny),
                        .out_dirs = new_out,
                    }
                );
            }
        }
    }

    // Identify any loaded cells in the active window that were not reached by the BFS
    u32 idx = 0u;
    for (i32 y = MIN_CHUNK_Y; y <= MAX_CHUNK_Y; ++y)
    {
        for (i32 dz = -rd; dz <= rd; ++dz)
        {
            for (i32 dx = -rd; dx <= rd; ++dx, ++idx)
            {
                const u32 state = snap[idx];
                if ((state & 0xFu) == GridCell::TAG_UNLOADED) continue;

                if (!visited.test(idx)) m_bfs_out_unwanted.emplace_back(origin.x + dx, y, origin.z + dz);
            }
        }
    }

    // Sort candidates closest first so near terrain and LOD transitions always complete before distant horizon chunks
    std::ranges::stable_sort(m_bfs_out_candidates, {}, [&](const glm::ivec3& c) { return distance_squared(c, origin); });
}

u8 WorldManager::compute_allowed_out_dirs(const glm::ivec3& coord, u32 idx, GridCell& cell, const glm::ivec3& origin) const
{
    const u32 portals = cell_portal_mask(idx);
    if (portals == 0u) return 0u;

    if (coord == origin)
    {
        cell.reached_faces = ALL_OPEN_FACES_MASK;
        return open_faces_mask(portals);
    }

    const u8 reached = static_cast<u8>(cell.reached_faces);
    if (reached == 0u) return 0u;

    u8 out = 0u;
    for (Direction in_dir : ALL_DIRECTIONS)
        if ((reached & direction_bit(in_dir)) != 0u) out |= reachable_faces_from(portals, in_dir);
    return out;
}

bool WorldManager::expand_frontier_from(const glm::ivec3& start_coord, const glm::ivec3& origin, i32 rd, bool high_priority)
{
    const u32 stamp = m_frontier_stamp;
    GridCell* const grid = m_grid.get();
    bool revealed_unloaded = false;

    const u32 start_idx = cell_index(start_coord);
    GridCell& root_cell = grid[start_idx];
    const u8 root_out = compute_allowed_out_dirs(start_coord, start_idx, root_cell, origin);
    if (root_out == 0u) return false;

    m_expand_stack.clear();
    m_expand_stack.push_back(
        SearchNode {
            .dx = static_cast<i16>(start_coord.x - origin.x),
            .dz = static_cast<i16>(start_coord.z - origin.z),
            .y = static_cast<i16>(start_coord.y),
            .out_dirs = root_out,
        }
    );

    while (!m_expand_stack.empty())
    {
        const SearchNode node = m_expand_stack.back();
        m_expand_stack.pop_back();

        const glm::ivec3 coord {origin.x + node.dx, node.y, origin.z + node.dz};

        for (Direction d : ALL_DIRECTIONS)
        {
            if ((node.out_dirs & direction_bit(d)) == 0u) continue;

            const glm::ivec3 ncoord = coord + to_offset(d);
            if (!is_y_in_world(ncoord.y) || std::abs(ncoord.x - origin.x) > rd || std::abs(ncoord.z - origin.z) > rd) continue;

            const u32 nidx = cell_index(ncoord);
            GridCell& ncell = grid[nidx];

            const Direction in_face = opposite(d);
            const u8 in_bit = direction_bit(in_face);
            const u8 prev_reached = static_cast<u8>(ncell.reached_faces);
            const bool newly_reached_face = (prev_reached & in_bit) == 0u;
            ncell.reached_faces = prev_reached | in_bit;

            const u8 target_lod = compute_lod(ncoord, origin);
            if (ncell.is_unloaded() || ncell.needs_lod_update(target_lod))
            {
                if (!high_priority && ncell.frontier_stamp == stamp) continue;

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
            else if (newly_reached_face)
            {
                const u32 n_portals = cell_portal_mask(nidx);
                const u8 n_out = reachable_faces_from(n_portals, in_face);
                if (n_out == 0u) continue;

                u8 prev_out = 0u;
                for (Direction prev_in : ALL_DIRECTIONS)
                    if ((prev_reached & direction_bit(prev_in)) != 0u) prev_out |= reachable_faces_from(n_portals, prev_in);

                const u8 new_out = static_cast<u8>(n_out & ~prev_out);
                if (new_out != 0u)
                {
                    m_expand_stack.push_back(
                        SearchNode {
                            .dx = static_cast<i16>(ncoord.x - origin.x),
                            .dz = static_cast<i16>(ncoord.z - origin.z),
                            .y = static_cast<i16>(ncoord.y),
                            .out_dirs = new_out,
                        }
                    );
                }
            }
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

        cell.reached_faces = ALL_OPEN_FACES_MASK;
        set_cell_air(idx, cell, air_lod);
        ++m_air_chunks_count;
        expand_frontier_from(c, origin, rd);
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

void WorldManager::start_background_gen(const glm::ivec3& origin)
{
    if (m_gen_in_flight) return;

    u32 batch_work = 0;
    m_gen_batch.clear();

    while (batch_work < GEN_WORK_BUDGET_PER_BATCH && m_gen_batch.size() < MAX_GEN_BATCH_CHUNKS)
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
        else
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

        const bool already_in_batch =
            std::ranges::any_of(m_gen_batch, [&](const GenTaskItem& existing) { return existing.coord == coord; });
        if (already_in_batch) continue;

        const u8 target_lod = compute_lod(coord, origin);
        const u32 idx = cell_index(coord);
        const GridCell& cell = m_grid[idx];

        if (!cell.is_unloaded() && !cell.needs_lod_update(target_lod)) continue;

        const bool target_full = m_pools.visit_lod(target_lod, [](const auto& pool) { return pool.full(); });
        if (target_full) continue;

        batch_work += lod_work_cost(target_lod);

        m_gen_batch.push_back(
            GenTaskItem {
                .coord = coord,
                .column_max_height = 0,
                .old_portal_mask = 0u,
                .portal_mask = 0u,
                .target_lod = target_lod,
                .was_unloaded = cell.is_unloaded(),
                .high_priority = from_urgent || !cell.is_unloaded(),
                .non_air = false,
                .guaranteed_all_lods = false,
                .committed = false,
                .staged_chunk = make_staged_chunk(target_lod, coord),
            }
        );
    }

    if (m_gen_batch.empty()) return;

    // Group chunks in the same column together so worker threads reuse 2D column noise
    std::ranges::sort(m_gen_batch, [](const GenTaskItem& a, const GenTaskItem& b) {
        if (a.target_lod != b.target_lod) return a.target_lod < b.target_lod;
        if (a.coord.x != b.coord.x) return a.coord.x < b.coord.x;
        if (a.coord.z != b.coord.z) return a.coord.z < b.coord.z;
        return a.coord.y < b.coord.y;
    });

    m_gen_in_flight = true;
    m_gen_task.m_SetSize = static_cast<u32>(m_gen_batch.size());
    m_gen_task.m_MinRange = std::max(1u, m_gen_task.m_SetSize / 4u);
    m_scheduler.AddTaskSetToPipe(&m_gen_task);
}

void WorldManager::execute_background_gen(enki::TaskSetPartition range, u32 threadnum)
{
    ChunkGenerator& gen = m_worker_generators[threadnum % m_worker_generators.size()];
    for (u32 i = range.start; i < range.end; ++i)
    {
        GenTaskItem& item = m_gen_batch[i];
        std::visit(
            [&](auto& chunk) {
                item.non_air = gen.generate(chunk, item.guaranteed_all_lods, item.column_max_height);
                if (item.non_air) item.portal_mask = extract_portal_mask(chunk);
            },
            item.staged_chunk
        );
    }
}

void WorldManager::harvest_background_gen(const glm::ivec3& origin, i32 rd, vulk::VulkanContext& ctx)
{
    m_gen_in_flight = false;

    // Pass 0: commit all generated chunks and portal masks to the grid before expanding frontiers
    for (GenTaskItem& item : m_gen_batch)
    {
        if (!is_in_active_window(item.coord)) continue;

        const u32 idx = cell_index(item.coord);
        GridCell& cell = m_grid[idx];
        const u8 current_target_lod = compute_lod(item.coord, origin);

        if (cell.is_real() && (cell.lod() == item.target_lod || cell.lod() == current_target_lod)) continue;
        if (cell.is_air() && item.target_lod >= cell.lod()) continue;

        item.was_unloaded = cell.is_unloaded();
        item.old_portal_mask = cell_portal_mask(idx);

        if (!item.non_air)
        {
            const u8 saved_reached = static_cast<u8>(cell.reached_faces);
            if (cell.is_real())
                unload_real_chunk(cell.lod(), cell.index, ctx);
            else if (cell.is_air())
                --m_air_chunks_count;

            const u8 air_lod = item.guaranteed_all_lods ? 1u : item.target_lod;
            cell.reached_faces = saved_reached;
            set_cell_air(idx, cell, air_lod);
            ++m_air_chunks_count;
            item.portal_mask = GridCell::ALL_PORTALS;
            item.committed = true;
        }
        else
        {
            const bool target_full = m_pools.visit_lod(item.target_lod, [](const auto& pool) { return pool.full(); });
            if (target_full) continue;

            const bool was_real = cell.is_real();
            bool had_old_mesh = false;
            if (cell.is_air()) --m_air_chunks_count;

            const u32 new_slot = m_pools.visit_lod(item.target_lod, [&](auto& new_pool) {
                using PoolType = std::remove_reference_t<decltype(new_pool)>;
                using ChunkType = Chunk<PoolType::LOD>;

                const u32 s = new_pool.allocate(item.coord);
                ChunkType& dst_chunk = new_pool.chunk_at(s);
                dst_chunk = std::move(std::get<ChunkType>(item.staged_chunk));
                dst_chunk.mesh_dirty = true;

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

            set_cell_real(idx, cell, item.target_lod, new_slot, item.portal_mask);
            if (item.high_priority || !item.was_unloaded)
                m_remesh_queues[item.target_lod - 1u].push_front(item.coord);
            else if (had_old_mesh)
                m_remesh_queues[item.target_lod - 1u].push_back(item.coord);
            else
                m_mesh_queues[item.target_lod - 1u].push_back(item.coord);

            item.committed = true;
        }

        // If the camera moved while this batch was generating so an even finer LOD is now needed, queue a follow-up
        if (cell.needs_lod_update(current_target_lod)) m_gen_queues[current_target_lod - 1u].push_front(item.coord);
    }

    // Pass 1: expand frontier for newly opened portals, fill sky air columns, and notify neighbors to remesh
    for (const GenTaskItem& item : m_gen_batch)
    {
        if (!item.committed || !is_in_active_window(item.coord)) continue;

        const u32 idx = cell_index(item.coord);
        GridCell& cell = m_grid[idx];
        const u32 newly_closed_portals = item.old_portal_mask & ~item.portal_mask;

        const u8 allowed_out = compute_allowed_out_dirs(item.coord, idx, cell, origin);
        const bool reaches_sky = ((allowed_out & direction_bit(Direction::Up)) != 0u) &&
                                 (!item.non_air || (((item.coord.y + 1) * CHUNK_SIZE) > item.column_max_height));

        if (item.was_unloaded && reaches_sky)
            fill_air_column_above(item.coord.x, item.coord.z, item.column_max_height, origin, rd);

        const bool expand_urgent = !item.was_unloaded;
        expand_frontier_from(item.coord, origin, rd, expand_urgent);

        if (newly_closed_portals != 0u)
        {
            m_search_dirty = true;
            if (m_bfs_in_flight) m_bfs_needs_rerun = true;
        }

        on_chunk_loaded(item.coord, item.high_priority || !item.was_unloaded);
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

void WorldManager::start_background_mesh()
{
    if (m_mesh_in_flight) return;

    m_mesh_batch.clear();
    u32 mesh_work = 0;

    while (mesh_work < MESH_WORK_BUDGET_PER_BATCH && m_mesh_batch.size() < MAX_MESH_BATCH_CHUNKS)
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

        const usize batch_idx = m_mesh_batch.size();
        MeshTaskItem item {
            .coord = coord,
            .slot = cell->index,
            .lod = cell->lod(),
            .neighbors = get_mesh_neighbors(coord),
            .staged_mesh = {},
        };
        std::swap(item.staged_mesh.vertices, m_mesh_vertex_scratch[batch_idx]);
        m_mesh_batch.push_back(std::move(item));
    }

    if (m_mesh_batch.empty()) return;

    m_mesh_in_flight = true;
    m_mesh_task.m_SetSize = static_cast<u32>(m_mesh_batch.size());
    m_mesh_task.m_MinRange = std::max(1u, m_mesh_task.m_SetSize / 4u);
    m_scheduler.AddTaskSetToPipe(&m_mesh_task);
}

void WorldManager::execute_background_mesh(enki::TaskSetPartition range, u32)
{
    for (u32 i = range.start; i < range.end; ++i)
    {
        MeshTaskItem& item = m_mesh_batch[i];
        m_pools.visit_lod(item.lod, [&](const auto& pool) {
            const auto& chunk = pool.chunk_at(item.slot);
            chunk.generate_mesh(item.staged_mesh, item.neighbors);
        });
    }
}

void WorldManager::harvest_background_mesh(vulk::VulkanContext& ctx)
{
    m_mesh_in_flight = false;

    u64 uploaded_bytes = 0;
    for (usize i = 0; i < m_mesh_batch.size(); ++i)
    {
        MeshTaskItem& item = m_mesh_batch[i];
        const GridCell* cell = find_real_cell(item.coord);

        if (cell != nullptr && cell->lod() == item.lod && cell->index == item.slot)
        {
            m_pools.visit_lod(item.lod, [&](auto& pool) {
                auto& chunk = pool.chunk_at(item.slot);
                ChunkMesh& mesh = pool.mesh_at(item.slot);
                const u64 mesh_bytes = item.staged_mesh.vertices.size() * sizeof(PackedVertex);

                if (uploaded_bytes > 0u && uploaded_bytes + mesh_bytes > MAX_UPLOAD_BYTES_PER_FRAME)
                {
                    chunk.mesh_dirty = true;
                    m_remesh_queues[item.lod - 1u].push_front(item.coord);
                    return;
                }

                std::swap(mesh.vertices, item.staged_mesh.vertices);
                mesh.quad_count = item.staged_mesh.quad_count;
                mesh.coord = item.coord;
                mesh.upload_to_gpu(ctx);
                uploaded_bytes += mesh_bytes;
                mesh.vertices.clear();
                std::swap(mesh.vertices, item.staged_mesh.vertices);
                mesh.meshed_once = true;
            });
        }

        item.staged_mesh.vertices.clear();
        std::swap(m_mesh_vertex_scratch[i], item.staged_mesh.vertices);
    }
}

} // namespace vl