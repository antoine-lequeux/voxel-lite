#pragma once

#include "chunkgenerator.hpp"
#include "containers.hpp"

#include <chrono>
#include <deque>

namespace vl
{

constexpr u32 MAX_ACTIVE_CHUNKS = 32768;

constexpr i32 MIN_CHUNK_Y = -8;
constexpr i32 MAX_CHUNK_Y = 31;
constexpr i32 WORLD_HEIGHT_CHUNKS = MAX_CHUNK_Y - MIN_CHUNK_Y + 1;

constexpr i32 MIN_RENDER_DIST = 4;
constexpr i32 MAX_RENDER_DIST = 64;
constexpr i32 DEFAULT_RENDER_DIST = 24;

constexpr i32 NEAR_RADIUS = 3;

// Per-frame budgets
constexpr f64 GEN_TIME_BUDGET_MS = 3.0;
constexpr f64 MESH_TIME_BUDGET_MS = 4.0;
constexpr u32 MAX_CHUNKS_GENERATED_PER_FRAME = 64;
constexpr u32 MAX_CHUNKS_MESHED_PER_FRAME = 64;
constexpr u32 MAX_AIR_CHUNKS_PER_FRAME = 16384;

class WorldManager
{
public:

    explicit WorldManager(i32 seed);

    [[nodiscard]] i32 get_render_distance() const { return m_render_distance; }
    void set_render_distance(i32 rd) { m_render_distance = std::clamp(rd, MIN_RENDER_DIST, MAX_RENDER_DIST); }

    [[nodiscard]] u32 get_active_chunks_count() const { return m_pool.size(); }
    [[nodiscard]] u32 get_air_chunks_count() const { return static_cast<u32>(m_air_chunks.size()); }
    [[nodiscard]] u32 get_pending_generation_count() const { return static_cast<u32>(m_gen_candidates.size()); }
    [[nodiscard]] static constexpr u32 get_max_chunks_capacity() { return MAX_ACTIVE_CHUNKS; }

    [[nodiscard]] auto get_active_chunks() { return m_pool.zipped(); }
    [[nodiscard]] auto get_active_chunks() const { return m_pool.zipped(); }

    [[nodiscard]] auto get_active_meshes() { return m_pool.active_meshes(); }
    [[nodiscard]] auto get_active_meshes() const { return m_pool.active_meshes(); }

    [[nodiscard]] static constexpr bool is_y_in_world(i32 cy) { return cy >= MIN_CHUNK_Y && cy <= MAX_CHUNK_Y; }

    [[nodiscard]] Chunk* get_chunk(const glm::ivec3& coord)
    {
        const u32 slot = find_real_slot(coord);
        return (slot != INVALID_POOL_INDEX) ? &m_pool.chunk_at(slot) : nullptr;
    }

    [[nodiscard]] const Chunk* get_chunk(const glm::ivec3& coord) const
    {
        const u32 slot = find_real_slot(coord);
        return (slot != INVALID_POOL_INDEX) ? &m_pool.chunk_at(slot) : nullptr;
    }

    [[nodiscard]] ChunkMesh* get_mesh(const glm::ivec3& coord)
    {
        const u32 slot = find_real_slot(coord);
        return (slot != INVALID_POOL_INDEX) ? &m_pool.mesh_at(slot) : nullptr;
    }

    // True if the chunk is loaded as an air chunk (no block data stored)
    [[nodiscard]] bool is_air_chunk(const glm::ivec3& coord) const { return find_air_index(coord) != INVALID_POOL_INDEX; }

    void set_world_block(i32 wx, i32 wy, i32 wz, BlockID id);

    // Raw neighbor pointers (nullptr for anything that is not a real loaded chunk)
    [[nodiscard]] ChunkNeighbors get_neighbors(const glm::ivec3& coord) const
    {
        return ChunkNeighbors {
            .left = get_chunk(coord + glm::ivec3(-1, 0, 0)),
            .right = get_chunk(coord + glm::ivec3(1, 0, 0)),
            .down = get_chunk(coord + glm::ivec3(0, -1, 0)),
            .up = get_chunk(coord + glm::ivec3(0, 1, 0)),
            .back = get_chunk(coord + glm::ivec3(0, 0, -1)),
            .front = get_chunk(coord + glm::ivec3(0, 0, 1)),
        };
    }

    bool unload_chunk(const glm::ivec3& coord, vulk::VulkanContext& ctx)
    {
        const u32 slot = find_real_slot(coord);
        if (slot != INVALID_POOL_INDEX)
        {
            unload_real_chunk(slot, ctx);
            return true;
        }

        const u32 air_idx = find_air_index(coord);
        if (air_idx != INVALID_POOL_INDEX)
        {
            remove_air_chunk(air_idx);
            return true;
        }
        return false;
    }

    void update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx);

private:

    using Clock = std::chrono::steady_clock;

    // Grid covering the maximum render window
    static constexpr i32 GRID_XZ = (2 * MAX_RENDER_DIST) + 1;
    static constexpr u32 GRID_CELLS = static_cast<u32>(GRID_XZ * GRID_XZ * WORLD_HEIGHT_CHUNKS);

    // Cell state encoding
    static constexpr u32 STATE_EMPTY = 0xFFFFFFFFu; // Not loaded
    static constexpr u32 AIR_BIT = 0x80000000u;     // Air chunk, low bits = index into m_air_chunks

    static_assert(MAX_ACTIVE_CHUNKS < AIR_BIT);

    [[nodiscard]] static constexpr bool is_real_state(u32 st) { return st < AIR_BIT; }
    [[nodiscard]] static constexpr bool is_air_state(u32 st) { return st != STATE_EMPTY && (st & AIR_BIT) != 0u; }

    static constexpr std::array<glm::ivec3, 6> DIR_OFFSETS {{
        {-1, 0, 0}, // Left
        {1, 0, 0},  // Right
        {0, -1, 0}, // Down
        {0, 1, 0},  // Up
        {0, 0, -1}, // Back
        {0, 0, 1},  // Front
    }};

    [[nodiscard]] static constexpr i32 wrap_xz(i32 v)
    {
        const i32 m = v % GRID_XZ;
        return (m < 0) ? m + GRID_XZ : m;
    }

    [[nodiscard]] static constexpr u32 cell_index(const glm::ivec3& c)
    {
        return static_cast<u32>((((c.y - MIN_CHUNK_Y) * GRID_XZ + wrap_xz(c.z)) * GRID_XZ) + wrap_xz(c.x));
    }

    [[nodiscard]] u32 find_real_slot(const glm::ivec3& coord) const
    {
        if (!is_y_in_world(coord.y)) return INVALID_POOL_INDEX;
        const u32 st = m_cell_state[cell_index(coord)];
        if (!is_real_state(st) || m_pool.chunk_at(st).coord != coord) return INVALID_POOL_INDEX;
        return st;
    }

    [[nodiscard]] u32 find_air_index(const glm::ivec3& coord) const
    {
        if (!is_y_in_world(coord.y)) return INVALID_POOL_INDEX;
        const u32 st = m_cell_state[cell_index(coord)];
        if (!is_air_state(st)) return INVALID_POOL_INDEX;
        const u32 idx = st & ~AIR_BIT;
        return (m_air_chunks[idx] == coord) ? idx : INVALID_POOL_INDEX;
    }

    // Fully opaque chunk used to replace unknown chunks during meshing
    [[nodiscard]] static const Chunk& solid_sentinel()
    {
        static const std::unique_ptr<Chunk> sentinel = [] {
            auto c = std::make_unique<Chunk>();
            c->fill(BlockType::Stone);
            c->recalculate_portal_mask();
            return c;
        }();
        return *sentinel;
    }

    [[nodiscard]] const Chunk* mesh_neighbor(const glm::ivec3& coord) const
    {
        if (!is_y_in_world(coord.y)) return nullptr; // Guaranteed empty

        const u32 st = m_cell_state[cell_index(coord)];
        if (is_real_state(st))
        {
            const Chunk& c = m_pool.chunk_at(st);
            return (c.coord == coord) ? &c : &solid_sentinel();
        }
        if (is_air_state(st) && m_air_chunks[st & ~AIR_BIT] == coord) return nullptr; // Air

        return &solid_sentinel(); // Unknown
    }

    [[nodiscard]] ChunkNeighbors get_mesh_neighbors(const glm::ivec3& coord) const
    {
        return ChunkNeighbors {
            .left = mesh_neighbor(coord + DIR_OFFSETS[0]),
            .right = mesh_neighbor(coord + DIR_OFFSETS[1]),
            .down = mesh_neighbor(coord + DIR_OFFSETS[2]),
            .up = mesh_neighbor(coord + DIR_OFFSETS[3]),
            .back = mesh_neighbor(coord + DIR_OFFSETS[4]),
            .front = mesh_neighbor(coord + DIR_OFFSETS[5]),
        };
    }

    [[nodiscard]] static constexpr u32 pack_entry(i32 dx, i32 y, i32 dz)
    {
        return static_cast<u32>(dx + MAX_RENDER_DIST) | (static_cast<u32>(dz + MAX_RENDER_DIST) << 8u) |
               (static_cast<u32>(y - MIN_CHUNK_Y) << 16u);
    }

    [[nodiscard]] u32 entry_cell(i32 dx, i32 y, i32 dz) const
    {
        const i32 wz = m_wrap_z[static_cast<usize>(dz + MAX_RENDER_DIST)];
        const i32 wx = m_wrap_x[static_cast<usize>(dx + MAX_RENDER_DIST)];
        return static_cast<u32>((((y - MIN_CHUNK_Y) * GRID_XZ + wz) * GRID_XZ) + wx);
    }

    // Mark every wanted cell and return the number of loaded chunks (real + air) that were reached.
    u32 run_search(const glm::ivec3& origin, i32 rd);

    [[nodiscard]] u32 allocate_real_chunk(const glm::ivec3& coord)
    {
        const u32 slot = m_pool.allocate(coord);
        m_cell_state[cell_index(coord)] = slot;
        m_slot_meshed[slot] = 0;
        m_mesh_queue.push_back(coord);
        return slot;
    }

    void unload_real_chunk(u32 slot, vulk::VulkanContext& ctx)
    {
        const glm::ivec3 coord = m_pool.chunk_at(slot).coord;
        m_cell_state[cell_index(coord)] = STATE_EMPTY;
        m_slot_meshed[slot] = 0;
        m_pool.free(slot, ctx);
    }

    void add_air_chunk(const glm::ivec3& coord)
    {
        const u32 idx = static_cast<u32>(m_air_chunks.size());
        m_air_chunks.push_back(coord);
        m_cell_state[cell_index(coord)] = AIR_BIT | idx;
    }

    void remove_air_chunk(u32 idx)
    {
        m_cell_state[cell_index(m_air_chunks[idx])] = STATE_EMPTY;

        const u32 last = static_cast<u32>(m_air_chunks.size()) - 1u;
        if (idx != last)
        {
            m_air_chunks[idx] = m_air_chunks[last];
            m_cell_state[cell_index(m_air_chunks[idx])] = AIR_BIT | idx;
        }
        m_air_chunks.pop_back();
    }

    // Returns true if any real chunk was unloaded
    bool unload_out_of_range(const glm::ivec3& origin, i32 rd, vulk::VulkanContext& ctx)
    {
        auto out_of_range = [&](const glm::ivec3& c) { return std::abs(c.x - origin.x) > rd || std::abs(c.z - origin.z) > rd; };

        bool freed = false;
        for (i32 i = static_cast<i32>(m_pool.size()) - 1; i >= 0; --i)
        {
            const u32 slot = m_pool.active_slot(static_cast<u32>(i));
            if (out_of_range(m_pool.chunk_at(slot).coord))
            {
                unload_real_chunk(slot, ctx);
                freed = true;
            }
        }

        for (i32 i = static_cast<i32>(m_air_chunks.size()) - 1; i >= 0; --i)
            if (out_of_range(m_air_chunks[static_cast<usize>(i)])) remove_air_chunk(static_cast<u32>(i));

        if (freed) m_search_dirty = true;
        return freed;
    }

    // Unloads every loaded chunk that was not marked as wanted by the last search. Returns true if any real chunk was freed.
    bool unload_unwanted(vulk::VulkanContext& ctx)
    {
        const u32 stamp = m_frame_stamp;
        bool freed = false;

        for (i32 i = static_cast<i32>(m_pool.size()) - 1; i >= 0; --i)
        {
            const u32 slot = m_pool.active_slot(static_cast<u32>(i));
            if (m_cell_stamp[cell_index(m_pool.chunk_at(slot).coord)] != stamp)
            {
                unload_real_chunk(slot, ctx);
                freed = true;
            }
        }

        for (i32 i = static_cast<i32>(m_air_chunks.size()) - 1; i >= 0; --i)
            if (m_cell_stamp[cell_index(m_air_chunks[static_cast<usize>(i)])] != stamp) remove_air_chunk(static_cast<u32>(i));

        return freed;
    }

    void generate_wanted_chunks(vulk::VulkanContext& ctx);

    // Queues meshed real neighbors for a priority remesh
    void on_chunk_loaded(const glm::ivec3& coord)
    {
        for (const glm::ivec3& o : DIR_OFFSETS)
        {
            const glm::ivec3 n = coord + o;
            const u32 slot = find_real_slot(n);
            if (slot == INVALID_POOL_INDEX || m_slot_meshed[slot] == 0u) continue;

            Chunk& c = m_pool.chunk_at(slot);
            if (c.mesh_dirty) continue; // Already queued

            c.mesh_dirty = true;
            m_remesh_queue.push_back(n);
        }
    }

    void mark_dirty(const glm::ivec3& coord)
    {
        const u32 slot = find_real_slot(coord);
        if (slot == INVALID_POOL_INDEX) return;

        Chunk& c = m_pool.chunk_at(slot);
        if (c.mesh_dirty) return; // Already queued

        c.mesh_dirty = true;
        m_remesh_queue.push_back(coord);
    }

    void process_mesh_queues(vulk::VulkanContext& ctx);

    // Spatial grid
    std::unique_ptr<u32[]> m_cell_state;
    std::unique_ptr<u32[]> m_cell_stamp;
    u32 m_frame_stamp = 0;
    std::array<i32, GRID_XZ> m_wrap_x {};
    std::array<i32, GRID_XZ> m_wrap_z {};

    // Loaded air chunks as coordinates
    std::vector<glm::ivec3> m_air_chunks;

    // Search state
    std::vector<u32> m_bfs_queue;
    std::vector<glm::ivec3> m_gen_candidates; // Wanted but not loaded
    std::vector<glm::ivec3> m_near_offsets;
    glm::ivec3 m_last_origin {0, 0, 0};
    i32 m_last_render_distance = 0;
    bool m_has_last_origin = false;
    bool m_search_dirty = true;

    // Meshing
    std::deque<glm::ivec3> m_remesh_queue; // Meshed chunks needing a remesh in priority
    std::deque<glm::ivec3> m_mesh_queue;   // Chunks never meshed
    std::unique_ptr<u8[]> m_slot_meshed;

    ChunkPool<MAX_ACTIVE_CHUNKS> m_pool {};
    i32 m_render_distance = DEFAULT_RENDER_DIST;
    ChunkGenerator m_generator {};
};

} // namespace vl