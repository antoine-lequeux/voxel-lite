#include "worldmanager.hpp"

#include <mimalloc.h>

namespace vl
{

WorldManager::WorldManager(i32 seed)
    : m_cell_state(std::make_unique<u32[]>(GRID_CELLS)), m_cell_stamp(std::make_unique<u32[]>(GRID_CELLS)),
      m_slot_meshed(std::make_unique<u8[]>(MAX_ACTIVE_CHUNKS)), m_generator(seed)
{
    std::fill_n(m_cell_state.get(), GRID_CELLS, STATE_EMPTY);
    std::fill_n(m_cell_stamp.get(), GRID_CELLS, 0u);
    std::fill_n(m_slot_meshed.get(), MAX_ACTIVE_CHUNKS, u8 {0});

    m_bfs_queue.reserve(GRID_CELLS);
    m_gen_candidates.reserve(16384);

    // Near-region offsets sorted by distance so the closest chunks are generated first
    for (i32 dy = -NEAR_RADIUS; dy <= NEAR_RADIUS; ++dy)
        for (i32 dz = -NEAR_RADIUS; dz <= NEAR_RADIUS; ++dz)
            for (i32 dx = -NEAR_RADIUS; dx <= NEAR_RADIUS; ++dx) m_near_offsets.emplace_back(dx, dy, dz);

    auto len2 = [](const glm::ivec3& v) { return (v.x * v.x) + (v.y * v.y) + (v.z * v.z); };
    std::stable_sort(m_near_offsets.begin(), m_near_offsets.end(), [&](const glm::ivec3& a, const glm::ivec3& b) {
        return len2(a) < len2(b);
    });
}

void WorldManager::set_world_block(i32 wx, i32 wy, i32 wz, BlockID id)
{
    // Floor division for power-of-two chunk size (CHUNK_SIZE = 2^P)
    constexpr auto P = std::countr_zero<u32>(CHUNK_SIZE);
    const glm::ivec3 chunk_coord {
        wx >> P,
        wy >> P,
        wz >> P,
    };

    if (!is_y_in_world(chunk_coord.y)) return;

    Chunk* chunk = get_chunk(chunk_coord);
    if (chunk == nullptr)
    {
        // Placing a block inside an air chunk turns it into a real chunk
        if (id == BlockType::Air || m_pool.full()) return;
        const u32 air_idx = find_air_index(chunk_coord);
        if (air_idx == INVALID_POOL_INDEX) return;

        remove_air_chunk(air_idx);
        const u32 slot = allocate_real_chunk(chunk_coord);
        chunk = &m_pool.chunk_at(slot);
        chunk->clear();
        chunk->recalculate_portal_mask();
    }

    const u32 lx = static_cast<u32>(wx & (CHUNK_SIZE - 1));
    const u32 ly = static_cast<u32>(wy & (CHUNK_SIZE - 1));
    const u32 lz = static_cast<u32>(wz & (CHUNK_SIZE - 1));

    chunk->set_block(lx, ly, lz, id);
    mark_dirty(chunk_coord);

    // Portals may have changed
    m_search_dirty = true;

    // If on a chunk boundary, mark the neighbors dirty as well
    constexpr u32 last = CHUNK_SIZE - 1u;
    if (lx == 0u) mark_dirty(chunk_coord + glm::ivec3(-1, 0, 0));
    if (lx == last) mark_dirty(chunk_coord + glm::ivec3(1, 0, 0));
    if (ly == 0u) mark_dirty(chunk_coord + glm::ivec3(0, -1, 0));
    if (ly == last) mark_dirty(chunk_coord + glm::ivec3(0, 1, 0));
    if (lz == 0u) mark_dirty(chunk_coord + glm::ivec3(0, 0, -1));
    if (lz == last) mark_dirty(chunk_coord + glm::ivec3(0, 0, 1));
}

void WorldManager::update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx)
{
    const i32 rd = m_render_distance;

    const glm::ivec3 cam_chunk {
        static_cast<i32>(std::floor(cam_world_pos.x / static_cast<f32>(CHUNK_SIZE))),
        static_cast<i32>(std::floor(cam_world_pos.y / static_cast<f32>(CHUNK_SIZE))),
        static_cast<i32>(std::floor(cam_world_pos.z / static_cast<f32>(CHUNK_SIZE))),
    };

    // The search origin is clamped vertically so a camera above/below the world still loads it
    const glm::ivec3 origin {cam_chunk.x, std::clamp(cam_chunk.y, MIN_CHUNK_Y, MAX_CHUNK_Y), cam_chunk.z};

    const bool window_changed =
        !m_has_last_origin || origin.x != m_last_origin.x || origin.z != m_last_origin.z || rd != m_last_render_distance;
    const bool origin_changed = window_changed || origin.y != m_last_origin.y;

    bool freed_real_chunks = false;

    // Out-of-range chunks must be unloaded before the search
    if (window_changed) freed_real_chunks |= unload_out_of_range(origin, rd, ctx);

    m_last_origin = origin;
    m_last_render_distance = rd;
    m_has_last_origin = true;

    if (origin_changed || m_search_dirty)
    {
        const u32 wanted_loaded = run_search(origin, rd);
        m_search_dirty = false;

        if (wanted_loaded != m_pool.size() + static_cast<u32>(m_air_chunks.size())) freed_real_chunks |= unload_unwanted(ctx);
    }

    if (freed_real_chunks) mi_collect(true);

    generate_wanted_chunks(ctx);
    process_mesh_queues(ctx);
}

u32 WorldManager::run_search(const glm::ivec3& origin, i32 rd)
{
    ++m_frame_stamp;
    const u32 stamp = m_frame_stamp;

    for (i32 i = 0; i < GRID_XZ; ++i)
    {
        m_wrap_x[static_cast<usize>(i)] = wrap_xz(origin.x + i - MAX_RENDER_DIST);
        m_wrap_z[static_cast<usize>(i)] = wrap_xz(origin.z + i - MAX_RENDER_DIST);
    }

    m_bfs_queue.clear();
    m_gen_candidates.clear();

    u32* const stamps = m_cell_stamp.get();
    const u32* const states = m_cell_state.get();

    // Every chunk of the near region is wanted regardless of portals
    for (const glm::ivec3& off : m_near_offsets)
    {
        const i32 y = origin.y + off.y;
        if (!is_y_in_world(y) || std::abs(off.x) > rd || std::abs(off.z) > rd) continue;

        const u32 cell = entry_cell(off.x, y, off.z);
        if (stamps[cell] == stamp) continue;
        stamps[cell] = stamp;
        m_bfs_queue.push_back(pack_entry(off.x, y, off.z));
    }

    u32 wanted_loaded = 0;

    for (usize head = 0; head < m_bfs_queue.size(); ++head)
    {
        const u32 e = m_bfs_queue[head];
        const i32 dx = static_cast<i32>(e & 0xFFu) - MAX_RENDER_DIST;
        const i32 dz = static_cast<i32>((e >> 8u) & 0xFFu) - MAX_RENDER_DIST;
        const i32 y = static_cast<i32>((e >> 16u) & 0xFFu) + MIN_CHUNK_Y;

        const u32 st = states[entry_cell(dx, y, dz)];

        // Not generated yet
        if (st == STATE_EMPTY)
        {
            m_gen_candidates.emplace_back(origin.x + dx, y, origin.z + dz);
            continue;
        }

        ++wanted_loaded;

        // Air chunks have all 6 portals open
        const bool is_air = (st & AIR_BIT) != 0u;
        const Chunk* chunk = is_air ? nullptr : &m_pool.chunk_at(st);

        for (u8 d = 0; d < 6; ++d)
        {
            if (!is_air && !chunk->has_portal(static_cast<Direction>(d))) continue;

            const glm::ivec3& o = DIR_OFFSETS[d];
            const i32 ndx = dx + o.x;
            const i32 ny = y + o.y;
            const i32 ndz = dz + o.z;

            if (!is_y_in_world(ny) || std::abs(ndx) > rd || std::abs(ndz) > rd) continue;

            const u32 ncell = entry_cell(ndx, ny, ndz);
            if (stamps[ncell] == stamp) continue;
            stamps[ncell] = stamp;
            m_bfs_queue.push_back(pack_entry(ndx, ny, ndz));
        }
    }

    return wanted_loaded;
}

void WorldManager::generate_wanted_chunks(vulk::VulkanContext& ctx)
{
    if (m_gen_candidates.empty()) return;

    const auto start = Clock::now();
    u32 real_count = 0;
    u32 air_count = 0;
    bool generated_any = false;

    for (const glm::ivec3& coord : m_gen_candidates)
    {
        const u32 cell = cell_index(coord);
        if (m_cell_state[cell] != STATE_EMPTY) continue; // Already generated earlier

        if (m_generator.is_chunk_empty(coord))
        {
            add_air_chunk(coord);
            on_chunk_loaded(coord);
            generated_any = true;
            if (++air_count >= MAX_AIR_CHUNKS_PER_FRAME) break;
        }
        else
        {
            if (m_pool.full()) continue; // Air chunks can still be discovered

            const u32 slot = allocate_real_chunk(coord);
            if (!m_generator.generate(m_pool.chunk_at(slot)))
            {
                // Turned out to be 100% air
                unload_real_chunk(slot, ctx);
                add_air_chunk(coord);
            }

            on_chunk_loaded(coord);
            generated_any = true;
            if (++real_count >= MAX_CHUNKS_GENERATED_PER_FRAME) break;
        }

        const f64 elapsed_ms = std::chrono::duration<f64, std::milli>(Clock::now() - start).count();
        if (elapsed_ms >= GEN_TIME_BUDGET_MS) break;
    }

    // Newly loaded chunks may open new paths for the search
    if (generated_any) m_search_dirty = true;
}

void WorldManager::process_mesh_queues(vulk::VulkanContext& ctx)
{
    const auto start = Clock::now();
    u32 meshed = 0;

    while (meshed < MAX_CHUNKS_MESHED_PER_FRAME)
    {
        glm::ivec3 coord {};
        if (!m_remesh_queue.empty())
        {
            coord = m_remesh_queue.front();
            m_remesh_queue.pop_front();
        }
        else if (!m_mesh_queue.empty())
        {
            coord = m_mesh_queue.front();
            m_mesh_queue.pop_front();
        }
        else
        {
            break;
        }

        // Skip stale entries (chunk unloaded or already up to date)
        const u32 slot = find_real_slot(coord);
        if (slot == INVALID_POOL_INDEX) continue;

        Chunk& chunk = m_pool.chunk_at(slot);
        if (!chunk.mesh_dirty) continue;

        ChunkMesh& mesh = m_pool.mesh_at(slot);
        chunk.generate_mesh(mesh, get_mesh_neighbors(coord));
        mesh.upload_to_gpu(ctx);
        mesh.vertices.clear();
        mesh.vertices.shrink_to_fit();
        chunk.mesh_dirty = false;
        m_slot_meshed[slot] = 1;
        ++meshed;

        const f64 elapsed_ms = std::chrono::duration<f64, std::milli>(Clock::now() - start).count();
        if (elapsed_ms >= MESH_TIME_BUDGET_MS) break;
    }
}

} // namespace vl