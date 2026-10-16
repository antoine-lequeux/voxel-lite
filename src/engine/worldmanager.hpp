#pragma once

#include "chunkgenerator.hpp"
#include "containers.hpp"
#include "renderer/vulkancontext.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vl
{

constexpr i32 RENDER_DIST = 24;
constexpr i32 LOAD_DIST = RENDER_DIST + 1;
constexpr i32 UNLOAD_DIST = RENDER_DIST + 2;

class WorldManager
{
public:

    static constexpr u32 MAX_ACTIVE_CHUNKS = 4096;
    static constexpr u32 HASH_MAP_CAPACITY = MAX_ACTIVE_CHUNKS * 2;

    [[nodiscard]] auto get_active_chunks() { return m_pool.zipped(); }
    [[nodiscard]] auto get_active_chunks() const { return m_pool.zipped(); }

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

        const u32 last_idx = m_pool.size() - 1u;
        const glm::ivec3 moved_coord = m_pool.swap_and_pop(idx, ctx);
        m_map.erase(coord);

        if (idx != last_idx) m_map.insert_or_assign(moved_coord, idx);
        return true;
    }

    void update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx)
    {
        const glm::ivec3 cam_pos {
            static_cast<i32>(std::floor(cam_world_pos.x / static_cast<f32>(CHUNK_SIZE))),
            0,
            static_cast<i32>(std::floor(cam_world_pos.z / static_cast<f32>(CHUNK_SIZE))),
        };

        // Unload unwanted chunks
        for (i32 i = static_cast<i32>(m_pool.size()) - 1; i >= 0; --i)
        {
            const glm::ivec3 c = m_pool.chunk_at(static_cast<u32>(i)).coord;
            if (std::abs(c.x - cam_pos.x) > UNLOAD_DIST || std::abs(c.z - cam_pos.z) > UNLOAD_DIST) unload_chunk(c, ctx);
        }

        // Gather missing chunks
        m_missing_scratchpad.clear();
        for (i32 x = cam_pos.x - LOAD_DIST; x <= cam_pos.x + LOAD_DIST; ++x)
        {
            for (i32 z = cam_pos.z - LOAD_DIST; z <= cam_pos.z + LOAD_DIST; ++z)
            {
                const glm::ivec3 coord {x, 0, z};
                if (m_map.find(coord) == INVALID_POOL_INDEX) m_missing_scratchpad.push_back(coord);
            }
        }

        // Sort radially from the camera center
        std::ranges::sort(m_missing_scratchpad, [&cam_world_pos](const glm::ivec3& a, const glm::ivec3& b) {
            constexpr f32 half = static_cast<f32>(CHUNK_SIZE) * 0.5f;
            const f32 ax = static_cast<f32>(a.x * CHUNK_SIZE) + half - cam_world_pos.x;
            const f32 az = static_cast<f32>(a.z * CHUNK_SIZE) + half - cam_world_pos.z;
            const f32 bx = static_cast<f32>(b.x * CHUNK_SIZE) + half - cam_world_pos.x;
            const f32 bz = static_cast<f32>(b.z * CHUNK_SIZE) + half - cam_world_pos.z;
            return (ax * ax + az * az) < (bx * bx + bz * bz);
        });

        // Generate up to 16 chunks per frame
        u32 gen_budget = 16;
        for (const glm::ivec3& coord : m_missing_scratchpad)
        {
            if (gen_budget == 0u || m_pool.full()) break;
            --gen_budget;

            const u32 idx = m_pool.allocate(coord);
            m_map.insert_or_assign(coord, idx);

            Chunk& new_chunk = m_pool.chunk_at(idx);
            ChunkGenerator::generate(new_chunk);

            // Flag existing neighbors so they are re-meshed
            mark_dirty(coord + glm::ivec3(-1, 0, 0));
            mark_dirty(coord + glm::ivec3(1, 0, 0));
            mark_dirty(coord + glm::ivec3(0, 0, -1));
            mark_dirty(coord + glm::ivec3(0, 0, 1));
        }

        // Mesh up to 2 dirty chunks per frame that are within load distance and have all neighbors loaded
        u32 mesh_budget = 2;
        std::span<Chunk> active_chunks = m_pool.chunks();
        std::span<ChunkMesh> active_meshes = m_pool.meshes();

        for (u32 i = 0; i < m_pool.size() && mesh_budget > 0u; ++i)
        {
            Chunk& chunk = active_chunks[i];
            if (!chunk.mesh_dirty) continue;

            if (std::abs(chunk.coord.x - cam_pos.x) <= LOAD_DIST && std::abs(chunk.coord.z - cam_pos.z) <= LOAD_DIST)
            {
                const ChunkNeighbors neighbors = get_neighbors(chunk.coord);
                if (neighbors.left && neighbors.right && neighbors.back && neighbors.front)
                {
                    chunk.generate_mesh(active_meshes[i], neighbors);
                    active_meshes[i].upload_to_gpu(ctx);
                    chunk.mesh_dirty = false;
                    --mesh_budget;
                }
            }
        }
    }

private:

    void mark_dirty(const glm::ivec3& coord)
    {
        if (Chunk* c = get_chunk(coord)) c->mesh_dirty = true;
    }

    FlatSpatialMap<HASH_MAP_CAPACITY> m_map {};
    ChunkPool<MAX_ACTIVE_CHUNKS> m_pool {};
    std::vector<glm::ivec3> m_missing_scratchpad {};
};

} // namespace vl