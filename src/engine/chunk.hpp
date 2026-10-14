#pragma once

#include "engine/block.hpp"
#include "renderer/vulkancontext.hpp"

#include <array>
#include <vector>

#include <glm/glm.hpp>

namespace vl
{

constexpr u32 CHUNK_SIZE = 32;
constexpr u32 CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE;
constexpr u32 CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

// Bit layout:
//   Bits  0..5  (6 bits) -> Local X position [0..32]
//   Bits  6..11 (6 bits) -> Local Y position [0..32]
//   Bits 12..17 (6 bits) -> Local Z position [0..32]
//   Bits 18..20 (3 bits) -> Face normal direction [0..5]
//   Bits 21..22 (2 bits) -> Per-vertex ambient occlusion level [0..3]
//   Bits 23..24 (2 bits) -> Face corner UV index [0..3] (bit 0 = U, bit 1 = V)
//   Bits 25..31 (7 bits) -> BlockID / material index [0..127]
using PackedVertex = u32;

[[nodiscard]] constexpr PackedVertex pack_vertex(u32 x, u32 y, u32 z, Direction normal, u32 ao_level, u32 uv_id, BlockID block_id)
{
    return (x & 0x3Fu) | ((y & 0x3Fu) << 6u) | ((z & 0x3Fu) << 12u) | ((static_cast<u32>(normal) & 0x07u) << 18u) |
           ((ao_level & 0x03u) << 21u) | ((uv_id & 0x03u) << 23u) | ((static_cast<u32>(block_id) & 0x7Fu) << 25u);
}

struct ChunkMesh
{
    std::vector<PackedVertex> vertices {};
    u32 quad_count = 0;

    // GPU storage buffer for vertex pulling
    vulk::BufferHandle vertex_buffer {};

    [[nodiscard]] constexpr u32 index_count() const { return quad_count * 6u; }
    [[nodiscard]] constexpr bool empty() const { return quad_count == 0u; }

    void clear()
    {
        vertices.clear();
        quad_count = 0;
    }

    // Uploads the generated PackedVertex array to VRAM and ensures the shared quad index buffer is large enough
    bool upload_to_gpu(vulk::VulkanContext& ctx);
    void destroy_gpu_resources(vulk::VulkanContext& ctx);
};

class Chunk;

// Neighbor pointers for cross-chunk face culling
struct ChunkNeighbors
{
    const Chunk* left = nullptr;  // -X
    const Chunk* right = nullptr; // +X
    const Chunk* down = nullptr;  // -Y
    const Chunk* up = nullptr;    // +Y
    const Chunk* back = nullptr;  // -Z
    const Chunk* front = nullptr; // +Z
};

class Chunk
{
public:

    // Flat array for contiguous memory
    std::array<BlockID, CHUNK_VOLUME> blocks {};

    // Chunk position in chunk-grid coordinates
    glm::ivec3 coord {0, 0, 0};

    [[nodiscard]] constexpr glm::vec3 world_origin() const
    {
        return glm::vec3 {
            static_cast<f32>(coord.x * static_cast<i32>(CHUNK_SIZE)),
            static_cast<f32>(coord.y * static_cast<i32>(CHUNK_SIZE)),
            static_cast<f32>(coord.z * static_cast<i32>(CHUNK_SIZE)),
        };
    }

    [[nodiscard]] static constexpr u32 get_index(u32 x, u32 y, u32 z) { return x + (y * CHUNK_SIZE) + (z * CHUNK_AREA); }

    [[nodiscard]] constexpr BlockID get_block_unchecked(u32 x, u32 y, u32 z) const { return blocks[get_index(x, y, z)]; }

    [[nodiscard]] constexpr BlockID get_block(u32 x, u32 y, u32 z) const
    {
        if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) return BlockType::Air;
        return blocks[get_index(x, y, z)];
    }

    constexpr void set_block(u32 x, u32 y, u32 z, BlockID id)
    {
        if (x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE) blocks[get_index(x, y, z)] = id;
    }

    constexpr void fill(BlockID id) { blocks.fill(id); }

    constexpr void clear() { blocks.fill(BlockType::Air); }

    [[nodiscard]] constexpr u32 count_non_air_blocks() const
    {
        u32 count = 0;
        for (const BlockID id : blocks)
            if (id != BlockType::Air) ++count;
        return count;
    }

    [[nodiscard]] BlockID sample_block(i32 x, i32 y, i32 z, const ChunkNeighbors& neighbors = {}) const;

    // Generates the visible surface mesh
    void generate_mesh(ChunkMesh& out_mesh, const ChunkNeighbors& neighbors = {}) const;
};

} // namespace vl