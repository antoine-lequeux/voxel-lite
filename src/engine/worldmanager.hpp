#pragma once

#include "chunkgenerator.hpp"
#include "containers.hpp"
#include "engine/chunk.hpp"
#include "renderer/vulkancontext.hpp"

#include <mimalloc.h>

#include <algorithm>
#include <cmath>

namespace vl
{

constexpr u32 MAX_ACTIVE_CHUNKS = 32768;
constexpr u32 HASH_MAP_CAPACITY = MAX_ACTIVE_CHUNKS * 2;

constexpr i32 MIN_RENDER_DIST = 2;
constexpr i32 MAX_RENDER_DIST = 64;
constexpr i32 DEFAULT_RENDER_DIST = 24;

constexpr i32 CHUNKS_LOADED_PER_FRAME = 16;
constexpr i32 CHUNKS_MESHED_PER_FRAME = 16;

class WorldManager
{
public:

    explicit WorldManager(i32 seed) : m_generator(seed) {}

    [[nodiscard]] i32 get_render_distance() const { return m_render_distance; }
    void set_render_distance(i32 rd) { m_render_distance = std::clamp(rd, MIN_RENDER_DIST, MAX_RENDER_DIST); }

    [[nodiscard]] u32 get_active_chunks_count() const { return m_pool.size(); }
    [[nodiscard]] static constexpr u32 get_max_chunks_capacity() { return MAX_ACTIVE_CHUNKS; }

    [[nodiscard]] auto get_active_chunks() { return m_pool.zipped(); }
    [[nodiscard]] auto get_active_chunks() const { return m_pool.zipped(); }

    [[nodiscard]] auto get_active_meshes() { return m_pool.active_meshes(); }
    [[nodiscard]] auto get_active_meshes() const { return m_pool.active_meshes(); }

    [[nodiscard]] Chunk* get_chunk(const glm::ivec3& coord)
    {
        const u32 idx = m_map.find(coord);
        return (idx != INVALID_POOL_INDEX) ? &m_pool.chunk_at(idx) : nullptr;
    }

    [[nodiscard]] const Chunk* get_chunk(const glm::ivec3& coord) const
    {
        const u32 idx = m_map.find(coord);
        return (idx != INVALID_POOL_INDEX) ? &m_pool.chunk_at(idx) : nullptr;
    }

    [[nodiscard]] ChunkMesh* get_mesh(const glm::ivec3& coord)
    {
        const u32 idx = m_map.find(coord);
        return (idx != INVALID_POOL_INDEX) ? &m_pool.mesh_at(idx) : nullptr;
    }

    void set_world_block(i32 wx, i32 wy, i32 wz, BlockID id)
    {
        // Floor division for power-of-two chunk size (CHUNK_SIZE = 2^P)
        constexpr auto P = std::countr_zero<u32>(CHUNK_SIZE);
        const glm::ivec3 chunk_coord {
            wx >> P,
            wy >> P,
            wz >> P,
        };

        Chunk* chunk = get_chunk(chunk_coord);
        if (chunk == nullptr) return;

        const u32 lx = static_cast<u32>(wx & 31);
        const u32 ly = static_cast<u32>(wy & 31);
        const u32 lz = static_cast<u32>(wz & 31);

        chunk->set_block(lx, ly, lz, id);
        chunk->mesh_dirty = true;

        // If on a chunk boundary, mark the adjacent neighbor dirty as well
        constexpr u32 last = CHUNK_SIZE - 1u;
        if (lx == 0u) mark_dirty(chunk_coord + glm::ivec3(-1, 0, 0));
        if (lx == last) mark_dirty(chunk_coord + glm::ivec3(1, 0, 0));
        if (ly == 0u) mark_dirty(chunk_coord + glm::ivec3(0, -1, 0));
        if (ly == last) mark_dirty(chunk_coord + glm::ivec3(0, 1, 0));
        if (lz == 0u) mark_dirty(chunk_coord + glm::ivec3(0, 0, -1));
        if (lz == last) mark_dirty(chunk_coord + glm::ivec3(0, 0, 1));
    }

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
        const u32 idx = m_map.find(coord);
        if (idx == INVALID_POOL_INDEX) return false;

        m_pool.free(idx, ctx);
        m_map.erase(coord);
        return true;
    }

    void update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx)
    {
        const i32 rd = m_render_distance;
        const i32 load_dist = rd + 1;
        const i32 unload_dist = rd + 2;

        const glm::ivec3 cam_pos {
            static_cast<i32>(std::floor(cam_world_pos.x / static_cast<f32>(CHUNK_SIZE))),
            static_cast<i32>(std::floor(cam_world_pos.y / static_cast<f32>(CHUNK_SIZE))),
            static_cast<i32>(std::floor(cam_world_pos.z / static_cast<f32>(CHUNK_SIZE))),
        };

        // Unload chunks outside unload_dist
        u32 unloaded_count = 0;
        for (i32 i = static_cast<i32>(m_pool.size()) - 1; i >= 0; --i)
        {
            const u32 slot = m_pool.active_slot(static_cast<u32>(i));
            const glm::ivec3 c = m_pool.chunk_at(slot).coord;
            if (std::abs(c.x - cam_pos.x) > unload_dist || std::abs(c.z - cam_pos.z) > unload_dist)
            {
                unload_chunk(c, ctx);
                ++unloaded_count;
            }
        }

        if (unloaded_count > 0) mi_collect(true);

        // Generate missing chunks radially outward in concentric square rings up to load_dist
        generate_missing_chunks_radially(cam_pos, load_dist, CHUNKS_LOADED_PER_FRAME);

        // Mesh dirty chunks that are within load_dist and have all neighbors
        u32 mesh_budget = CHUNKS_MESHED_PER_FRAME;
        const u32 total_active = m_pool.size();

        for (u32 i = 0; i < total_active && mesh_budget > 0u; ++i)
        {
            const u32 slot = m_pool.active_slot(i);
            Chunk& chunk = m_pool.chunk_at(slot);
            if (!chunk.mesh_dirty) continue;

            if (std::abs(chunk.coord.x - cam_pos.x) <= load_dist && std::abs(chunk.coord.z - cam_pos.z) <= load_dist)
            {
                const ChunkNeighbors neighbors = get_neighbors(chunk.coord);
                if (neighbors.left && neighbors.right && neighbors.back && neighbors.front)
                {
                    ChunkMesh& mesh = m_pool.mesh_at(slot);
                    chunk.generate_mesh(mesh, neighbors);
                    mesh.upload_to_gpu(ctx);
                    mesh.vertices.clear();
                    mesh.vertices.shrink_to_fit();
                    chunk.mesh_dirty = false;
                    --mesh_budget;
                }
            }
        }
    }

private:

    void generate_missing_chunks_radially(const glm::ivec3& cam_pos, i32 max_radius, u32 max_chunks_to_gen)
    {
        u32 gen_budget = max_chunks_to_gen;

        // Camera chunk
        const glm::ivec3 center {cam_pos.x, 0, cam_pos.z};
        if (m_map.find(center) == INVALID_POOL_INDEX)
        {
            allocate_and_generate(center);
            if (--gen_budget == 0u || m_pool.full()) return;
        }

        // Concentric square rings outward
        for (i32 r = 1; r <= max_radius; ++r)
        {
            for (i32 dx = -r; dx <= r; ++dx)
            {
                const glm::ivec3 top {cam_pos.x + dx, 0, cam_pos.z + r};
                if (m_map.find(top) == INVALID_POOL_INDEX)
                {
                    allocate_and_generate(top);
                    if (--gen_budget == 0u || m_pool.full()) return;
                }

                const glm::ivec3 bottom {cam_pos.x + dx, 0, cam_pos.z - r};
                if (m_map.find(bottom) == INVALID_POOL_INDEX)
                {
                    allocate_and_generate(bottom);
                    if (--gen_budget == 0u || m_pool.full()) return;
                }
            }
            for (i32 dz = -r + 1; dz <= r - 1; ++dz)
            {
                const glm::ivec3 left {cam_pos.x - r, 0, cam_pos.z + dz};
                if (m_map.find(left) == INVALID_POOL_INDEX)
                {
                    allocate_and_generate(left);
                    if (--gen_budget == 0u || m_pool.full()) return;
                }

                const glm::ivec3 right {cam_pos.x + r, 0, cam_pos.z + dz};
                if (m_map.find(right) == INVALID_POOL_INDEX)
                {
                    allocate_and_generate(right);
                    if (--gen_budget == 0u || m_pool.full()) return;
                }
            }
        }
    }

    void allocate_and_generate(const glm::ivec3& coord)
    {
        const u32 idx = m_pool.allocate(coord);
        m_map.insert_or_assign(coord, idx);

        Chunk& new_chunk = m_pool.chunk_at(idx);
        m_generator.generate(new_chunk);

        mark_dirty(coord + glm::ivec3(-1, 0, 0));
        mark_dirty(coord + glm::ivec3(1, 0, 0));
        mark_dirty(coord + glm::ivec3(0, 0, -1));
        mark_dirty(coord + glm::ivec3(0, 0, 1));
    }

    void mark_dirty(const glm::ivec3& coord)
    {
        if (Chunk* c = get_chunk(coord)) c->mesh_dirty = true;
    }

    FlatSpatialMap<HASH_MAP_CAPACITY> m_map {};
    ChunkPool<MAX_ACTIVE_CHUNKS> m_pool {};
    i32 m_render_distance = DEFAULT_RENDER_DIST;
    ChunkGenerator m_generator {};
};

} // namespace vl