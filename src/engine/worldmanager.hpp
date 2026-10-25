#pragma once

#include "chunkgenerator.hpp"
#include "containers.hpp"

#include <TaskScheduler.h>

#include <chrono>
#include <deque>

namespace vl
{

inline constexpr i32 NEAR_RADIUS = 3;

// Per-frame budgets
inline constexpr f64 SKY_AIR_TIME_BUDGET_MS = 2.0;
inline constexpr f64 BFS_MIN_INTERVAL_MS = 250.0;
inline constexpr u32 GEN_WORK_BUDGET_PER_FRAME = 4096;
inline constexpr u32 MESH_WORK_BUDGET_PER_FRAME = 4096;
inline constexpr u32 MAX_GEN_BATCH_CHUNKS = 2048;
inline constexpr u32 MAX_MESH_BATCH_CHUNKS = 2048;
inline constexpr u64 MAX_UPLOAD_BYTES_PER_FRAME = 16ull * 2048 * 2048;

[[nodiscard]] constexpr i32 length_squared(const glm::ivec3& v)
{
    return (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
}

[[nodiscard]] constexpr i32 distance_squared(const glm::ivec3& a, const glm::ivec3& b)
{
    return length_squared(a - b);
}
class WorldManager
{
public:

    explicit WorldManager(i32 seed);
    ~WorldManager();

    WorldManager(const WorldManager&) = delete;
    WorldManager& operator=(const WorldManager&) = delete;
    WorldManager(WorldManager&&) = delete;
    WorldManager& operator=(WorldManager&&) = delete;

    [[nodiscard]] i32 get_render_distance() const { return m_render_distance; }
    void set_render_distance(i32 rd) { m_render_distance = std::clamp(rd, MIN_RENDER_DIST, MAX_RENDER_DIST); }

    [[nodiscard]] u32 get_active_chunks_count() const { return m_pools.total_active_chunks(); }
    [[nodiscard]] u32 get_air_chunks_count() const { return m_air_chunks_count; }
    [[nodiscard]] u32 get_pending_generation_count() const
    {
        u32 total = 0;
        for (const auto& q : m_gen_queues) total += static_cast<u32>(q.size());
        return total;
    }
    [[nodiscard]] static constexpr u32 get_max_chunks_capacity() { return LodPoolGroup::total_capacity(); }

    template <u32 Lod = 1>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    [[nodiscard]] u32 get_lod_chunks_count() const
    {
        return m_pools.pool<Lod>().size();
    }

    template <typename Fn>
    void for_each_active_mesh(Fn&& fn) const
    {
        m_pools.for_each_active_mesh(std::forward<Fn>(fn));
    }

    void destroy_all_gpu_resources(vulk::VulkanContext& ctx);

    [[nodiscard]] static constexpr bool is_y_in_world(i32 cy) { return cy >= MIN_CHUNK_Y && cy <= MAX_CHUNK_Y; }

    // Computes which LOD a chunk belongs to
    [[nodiscard]] static constexpr u8 compute_lod(const glm::ivec3& coord, const glm::ivec3& origin)
    {
        const i32 dist = std::max({
            std::abs(coord.x - origin.x),
            std::abs(coord.y - origin.y),
            std::abs(coord.z - origin.z),
        });

        for (u32 lod = 1; lod <= MAX_LOD; ++lod)
            if (dist <= LOD_BOX_RADII[lod - 1u]) return static_cast<u8>(lod);
        return static_cast<u8>(MAX_LOD);
    }

    // Estimated CPU work weight of generating or meshing a chunk
    [[nodiscard]] static constexpr u32 lod_work_cost(u8 lod)
    {
        switch (lod)
        {
            case 1: return 32u;
            case 2: return 16u;
            case 3: return 8u;
            case 4: return 2u;
            default: return 1u;
        }
    }

    template <u32 Lod = 1>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    [[nodiscard]] Chunk<Lod>* get_chunk(const glm::ivec3& coord)
    {
        const GridCell* cell = find_real_cell(coord);
        if (cell == nullptr || cell->lod() != Lod) return nullptr;
        return &m_pools.pool<Lod>().chunk_at(cell->index);
    }

    template <u32 Lod = 1>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    [[nodiscard]] const Chunk<Lod>* get_chunk(const glm::ivec3& coord) const
    {
        const GridCell* cell = find_real_cell(coord);
        if (cell == nullptr || cell->lod() != Lod) return nullptr;
        return &m_pools.pool<Lod>().chunk_at(cell->index);
    }

    [[nodiscard]] ChunkMesh* get_mesh(const glm::ivec3& coord)
    {
        const GridCell* cell = find_real_cell(coord);
        if (cell == nullptr) return nullptr;
        return m_pools.visit_lod(cell->lod(), [&](auto& pool) { return &pool.mesh_at(cell->index); });
    }

    [[nodiscard]] bool is_air_chunk(const glm::ivec3& coord) const
    {
        if (!is_in_active_window(coord)) return false;
        return m_grid[cell_index(coord)].is_air();
    }

    void set_world_block(i32 wx, i32 wy, i32 wz, BlockID id);

    bool unload_chunk(const glm::ivec3& coord, vulk::VulkanContext& ctx);

    void update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx);

private:

    using Clock = std::chrono::steady_clock;

    // Grid covering the maximum render window
    static constexpr i32 GRID_XZ = (2 * MAX_RENDER_DIST) + 1;
    static constexpr u32 GRID_CELLS = static_cast<u32>(GRID_XZ * GRID_XZ * WORLD_HEIGHT_CHUNKS);
    static constexpr u32 BITSET_WORDS = (GRID_CELLS + 63u) / 64u;

    struct GridCell
    {
        static constexpr u32 TAG_UNLOADED = 0u;
        static constexpr u32 ALL_PORTALS = 0b00111111u;

        // Tag encoding (4 bits):
        //   0                            = unloaded
        //   1..MAX_LOD                   = real chunk at LOD 'tag'
        //   (MAX_LOD + 1)..(2 * MAX_LOD) = air chunk evaluated at LOD 'tag - MAX_LOD' (1 = guaranteed air at all LODs)
        u32 tag : 4 = TAG_UNLOADED;
        u32 portal_mask : 6 = 0u;
        u32 index : 22 = 0u;
        u32 loaded_epoch = 0u;
        u32 frontier_stamp = 0u;

        [[nodiscard]] constexpr bool is_unloaded() const { return tag == TAG_UNLOADED; }
        [[nodiscard]] constexpr bool is_real() const { return tag >= 1u && tag <= MAX_LOD; }
        [[nodiscard]] constexpr bool is_air() const { return tag > MAX_LOD; }
        [[nodiscard]] constexpr u8 lod() const { return static_cast<u8>(tag <= MAX_LOD ? tag : (tag - MAX_LOD)); }

        [[nodiscard]] constexpr bool needs_lod_update(u8 target_lod) const
        {
            if (is_real()) return lod() != target_lod;
            if (is_air()) return target_lod < lod();
            return false;
        }

        [[nodiscard]] constexpr u16 pack_bfs_state() const { return static_cast<u16>((portal_mask << 4u) | tag); }
    };

    // BFS queue entry supporting render distances up to 2047 chunks
    struct SearchNode
    {
        i32 dx : 12;
        i32 dz : 12;
        i32 y : 8;
    };

    static_assert(sizeof(SearchNode) == 4);

    struct GenTaskItem
    {
        glm::ivec3 coord {};
        u32 slot = 0;
        i32 column_max_height = 0;
        u8 target_lod = 1;
        u8 old_portal_mask = 0;
        bool was_unloaded = false;
        bool was_real = false;
        bool had_old_mesh = false;
        bool high_priority = false;
        bool non_air = false;
        bool guaranteed_all_lods = false;
        u8 portal_mask = 0;
    };

    struct MeshTaskItem
    {
        glm::ivec3 coord {};
        u32 slot = 0;
        u8 lod = 1;
        ChunkNeighbors neighbors {};
    };

    struct BackgroundBfsTask : enki::ITaskSet
    {
        WorldManager* manager = nullptr;

        explicit BackgroundBfsTask(WorldManager* mgr) : enki::ITaskSet(1), manager(mgr) { m_Priority = enki::TASK_PRIORITY_LOW; }

        void ExecuteRange(enki::TaskSetPartition, u32) override { manager->execute_background_bfs(); }
    };

    [[nodiscard]] static constexpr i32 wrap_xz(i32 v)
    {
        const i32 m = v % GRID_XZ;
        return (m < 0) ? m + GRID_XZ : m;
    }

    [[nodiscard]] static constexpr u32 cell_index(const glm::ivec3& c)
    {
        return static_cast<u32>((((c.y - MIN_CHUNK_Y) * GRID_XZ + wrap_xz(c.z)) * GRID_XZ) + wrap_xz(c.x));
    }

    [[nodiscard]] bool is_in_active_window(const glm::ivec3& c) const
    {
        if (!m_has_last_origin || !is_y_in_world(c.y)) return false;
        return std::abs(c.x - m_last_origin.x) <= m_last_render_distance &&
               std::abs(c.z - m_last_origin.z) <= m_last_render_distance;
    }

    void clear_cell(u32 idx, GridCell& cell)
    {
        cell.tag = GridCell::TAG_UNLOADED;
        cell.portal_mask = 0u;
        cell.index = 0u;
        m_cell_bfs_state[idx] = 0u;
    }

    void set_cell_air(u32 idx, GridCell& cell, u8 air_lod)
    {
        cell.tag = static_cast<u32>(MAX_LOD + air_lod);
        cell.portal_mask = GridCell::ALL_PORTALS;
        cell.index = 0u;
        cell.loaded_epoch = ++m_current_epoch;
        m_cell_bfs_state[idx] = cell.pack_bfs_state();
    }

    void set_cell_real(u32 idx, GridCell& cell, u8 chunk_lod, u32 pool_slot, u8 portals)
    {
        cell.tag = chunk_lod;
        cell.portal_mask = portals & GridCell::ALL_PORTALS;
        cell.index = pool_slot;
        cell.loaded_epoch = ++m_current_epoch;
        m_cell_bfs_state[idx] = cell.pack_bfs_state();
    }

    void update_cell_portals(u32 idx, GridCell& cell, u8 portals)
    {
        cell.portal_mask = portals & GridCell::ALL_PORTALS;
        cell.loaded_epoch = ++m_current_epoch;
        m_cell_bfs_state[idx] = cell.pack_bfs_state();
    }

    [[nodiscard]] const GridCell* find_real_cell(const glm::ivec3& coord) const
    {
        if (!is_in_active_window(coord)) return nullptr;
        const GridCell& cell = m_grid[cell_index(coord)];
        if (!cell.is_real()) return nullptr;
        return &cell;
    }

    [[nodiscard]] NeighborChunk mesh_neighbor(const glm::ivec3& coord) const
    {
        if (!is_y_in_world(coord.y)) return NeighborChunk::air();
        if (!is_in_active_window(coord)) return NeighborChunk::solid();

        const GridCell& cell = m_grid[cell_index(coord)];
        if (cell.is_real())
        {
            return m_pools.visit_lod(cell.lod(), [&](const auto& pool) -> NeighborChunk {
                return NeighborChunk::from_chunk(pool.chunk_at(cell.index));
            });
        }

        if (cell.is_air()) return NeighborChunk::air();

        return NeighborChunk::solid();
    }

    [[nodiscard]] ChunkNeighbors get_mesh_neighbors(const glm::ivec3& coord) const
    {
        return ChunkNeighbors {
            .left = mesh_neighbor(coord + to_offset(Direction::Left)),
            .right = mesh_neighbor(coord + to_offset(Direction::Right)),
            .down = mesh_neighbor(coord + to_offset(Direction::Down)),
            .up = mesh_neighbor(coord + to_offset(Direction::Up)),
            .back = mesh_neighbor(coord + to_offset(Direction::Back)),
            .front = mesh_neighbor(coord + to_offset(Direction::Front)),
        };
    }

    template <typename ChunkT>
    [[nodiscard]] static constexpr u8 extract_portal_mask(const ChunkT& chunk)
    {
        u8 mask = 0;
        for (Direction dir : ALL_DIRECTIONS)
            if (chunk.has_portal(dir)) mask |= direction_bit(dir);
        return mask;
    }

    void start_background_bfs(const glm::ivec3& origin, i32 rd);
    void harvest_background_bfs(const glm::ivec3& origin, vulk::VulkanContext& ctx);
    void execute_background_bfs();

    // Returns true if at least one unloaded neighbor was newly queued
    bool expand_frontier_from(
        const glm::ivec3& coord, const GridCell& start_cell, const glm::ivec3& origin, i32 rd, bool high_priority = false
    );

    // Fills all unloaded air chunks in a column strictly above 'col_max_height' and expands their frontier
    void fill_air_column_above(i32 cx, i32 cz, i32 col_max_height, const glm::ivec3& origin, i32 rd);

    void unload_real_chunk(u8 lod, u32 slot, vulk::VulkanContext& ctx);
    void unload_cell_at(const glm::ivec3& coord, vulk::VulkanContext& ctx);
    void unload_out_of_range(const glm::ivec3& new_origin, i32 new_rd, vulk::VulkanContext& ctx);

    void generate_wanted_chunks(const glm::ivec3& origin, i32 rd, vulk::VulkanContext& ctx);
    void on_chunk_loaded(const glm::ivec3& coord, bool high_priority = false);
    void mark_dirty(const glm::ivec3& coord);
    void process_mesh_queues(vulk::VulkanContext& ctx);

    // Spatial grid (main thread)
    std::unique_ptr<GridCell[]> m_grid {};
    std::unique_ptr<u16[]> m_cell_bfs_state {};
    u32 m_current_epoch = 0;
    u32 m_frontier_stamp = 1;
    u32 m_air_chunks_count = 0;

    // Background BFS state (isolated from main thread)
    BackgroundBfsTask m_bfs_task;
    std::unique_ptr<u16[]> m_bfs_snapshot {};
    std::unique_ptr<u64[]> m_bfs_visited_bits {};
    std::vector<SearchNode> m_bfs_queue {};
    std::vector<glm::ivec3> m_bfs_out_candidates {};
    std::vector<glm::ivec3> m_bfs_out_unwanted {};
    glm::ivec3 m_bfs_origin {0, 0, 0};
    i32 m_bfs_rd = 0;
    u32 m_bfs_start_epoch = 0;
    bool m_bfs_in_flight = false;
    bool m_bfs_needs_rerun = false;
    Clock::time_point m_last_bfs_time {};

    // Per-LOD generation and meshing queues (index 0 = LOD 1, index MAX_LOD-1 = LOD 5)
    std::deque<glm::ivec3> m_urgent_gen_queue {};
    std::array<std::deque<glm::ivec3>, MAX_LOD> m_gen_queues {};
    std::vector<glm::ivec3> m_frontier_during_bfs {};
    std::vector<glm::ivec3> m_near_offsets {};
    glm::ivec3 m_last_origin {0, 0, 0};
    i32 m_last_render_distance = 0;
    bool m_has_last_origin = false;
    bool m_search_dirty = true;

    std::array<std::deque<glm::ivec3>, MAX_LOD> m_remesh_queues {};
    std::array<std::deque<glm::ivec3>, MAX_LOD> m_mesh_queues {};

    // Multithreaded worker state
    enki::TaskScheduler m_scheduler {};
    std::vector<ChunkGenerator> m_worker_generators {};
    std::vector<GenTaskItem> m_gen_batch {};
    std::vector<MeshTaskItem> m_mesh_batch {};

    // Per-LOD chunk pools
    LodPoolGroup m_pools {};
    i32 m_render_distance = DEFAULT_RENDER_DIST;
};

} // namespace vl
