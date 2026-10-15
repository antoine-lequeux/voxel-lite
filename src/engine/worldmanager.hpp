#pragma once

#include "chunk.hpp"
#include "chunkgenerator.hpp"
#include "engine/chunk.hpp"
#include "renderer/vulkancontext.hpp"

#include <algorithm>
#include <deque>
#include <ranges>
#include <unordered_map>

namespace vl
{

struct IVec3Hash
{
    [[nodiscard]] constexpr usize operator()(const glm::ivec3& v) const
    {
        // Prime multiplication hash
        return (static_cast<usize>(v.x) * 73856093ull) ^ (static_cast<usize>(v.y) * 19349663ull) ^
               (static_cast<usize>(v.z) * 83492791ull);
    }
};

constexpr i32 RD = 16;

class WorldManager
{
public:

    [[nodiscard]] auto get_active_chunks() { return std::views::zip(m_chunk_pool, m_mesh_pool); }
    [[nodiscard]] auto get_active_chunks() const { return std::views::zip(m_chunk_pool, m_mesh_pool); }

    void update(const glm::vec3& cam_world_pos, vulk::VulkanContext& ctx)
    {
        glm::ivec3 cam_pos = {
            static_cast<i32>(std::floor(cam_world_pos.x / CHUNK_SIZE)), 0,
            static_cast<i32>(std::floor(cam_world_pos.z / CHUNK_SIZE))
        };

        constexpr i32 GEN_RD = RD + 1;

        // Collect all missing chunks
        std::vector<glm::ivec3> missing_chunks;
        for (i32 x = cam_pos.x - GEN_RD; x <= cam_pos.x + GEN_RD; x++)
        {
            for (i32 z = cam_pos.z - GEN_RD; z <= cam_pos.z + GEN_RD; z++)
            {
                glm::ivec3 coord {x, 0, z};
                if (!get_chunk(coord)) missing_chunks.push_back(coord);
            }
        }

        // Sort missing chunks so those closest to the player generate first
        std::ranges::sort(missing_chunks, [&cam_world_pos](const glm::ivec3& a, const glm::ivec3& b) {
            f32 dx_a = (static_cast<f32>(a.x * CHUNK_SIZE)) - cam_world_pos.x;
            f32 dz_a = (static_cast<f32>(a.z * CHUNK_SIZE)) - cam_world_pos.z;
            f32 dx_b = (static_cast<f32>(b.x * CHUNK_SIZE)) - cam_world_pos.x;
            f32 dz_b = (static_cast<f32>(b.z * CHUNK_SIZE)) - cam_world_pos.z;
            return (dx_a * dx_a + dz_a * dz_a) < (dx_b * dx_b + dz_b * dz_b);
        });

        // Generate 2 chunks per frame
        i32 gen_limit = 2;
        for (const glm::ivec3& coord : missing_chunks)
        {
            if (gen_limit-- == 0) break;

            usize idx = m_chunk_pool.size();
            m_chunk_map[coord] = idx;
            m_chunk_pool.emplace_back();
            Chunk& new_chunk = m_chunk_pool.back();
            new_chunk.coord = coord;
            new_chunk.mesh_dirty = true; // Mark for meshing

            ChunkGenerator::generate(new_chunk);
            m_mesh_pool.emplace_back();

            // Chunks must be remeshed when a neighbor is generated
            auto mark_dirty = [this](const glm::ivec3& c) {
                if (Chunk* neighbor = get_chunk(c)) neighbor->mesh_dirty = true;
            };
            mark_dirty(coord + glm::ivec3(-1, 0, 0));
            mark_dirty(coord + glm::ivec3(1, 0, 0));
            mark_dirty(coord + glm::ivec3(0, 0, -1));
            mark_dirty(coord + glm::ivec3(0, 0, 1));
        }

        // Mesh 2 dirty chunks per frame
        i32 mesh_limit = 2;
        for (usize i = 0; i < m_chunk_pool.size(); ++i)
        {
            if (mesh_limit == 0) break;

            Chunk& chunk = m_chunk_pool[i];

            if (chunk.mesh_dirty)
            {
                ChunkMesh& mesh = m_mesh_pool[i];
                chunk.generate_mesh(mesh, get_neighbors(chunk.coord));
                mesh.upload_to_gpu(ctx);
                chunk.mesh_dirty = false;
                mesh_limit--;
            }
        }
    }

    [[nodiscard]] Chunk* get_chunk(const glm::ivec3& coord)
    {
        auto it = m_chunk_map.find(coord);
        return (it != m_chunk_map.end()) ? &m_chunk_pool[it->second] : nullptr;
    }

    [[nodiscard]] ChunkNeighbors get_neighbors(const glm::ivec3& coord)
    {
        return ChunkNeighbors {
            .left = get_chunk(coord + glm::ivec3(-1, 0, 0)),
            .right = get_chunk(coord + glm::ivec3(1, 0, 0)),
            .down = get_chunk(coord + glm::ivec3(0, -1, 0)),
            .up = get_chunk(coord + glm::ivec3(0, 1, 0)),
            .back = get_chunk(coord + glm::ivec3(0, 0, -1)),
            .front = get_chunk(coord + glm::ivec3(0, 0, 1))
        };
    }

private:

    // Maps a 3D coordinate to an index in the pools
    std::unordered_map<glm::ivec3, usize, IVec3Hash> m_chunk_map;

    std::deque<Chunk> m_chunk_pool;
    std::deque<ChunkMesh> m_mesh_pool;
};

} // namespace vl