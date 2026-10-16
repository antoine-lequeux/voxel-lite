#pragma once

#include "engine/block.hpp"
#include "renderer/vulkancontext.hpp"

#include <array>
#include <vector>

#include <glm/glm.hpp>

namespace vl
{

constexpr i32 CHUNK_SIZE = 32;
constexpr i32 CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE;
constexpr i32 CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

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
    glm::ivec3 coord {0, 0, 0};
    std::vector<PackedVertex> vertices {};
    u32 quad_count = 0;

    // GPU storage buffer for vertex pulling
    vulk::BufferHandle vertex_buffer {};

    [[nodiscard]] constexpr u32 index_count() const { return quad_count * 6u; }
    [[nodiscard]] constexpr bool empty() const { return quad_count == 0u; }
    [[nodiscard]] constexpr glm::vec3 world_origin() const
    {
        return glm::vec3 {
            static_cast<f32>(coord.x * CHUNK_SIZE),
            static_cast<f32>(coord.y * CHUNK_SIZE),
            static_cast<f32>(coord.z * CHUNK_SIZE),
        };
    }

    void clear()
    {
        vertices.clear();
        vertices.shrink_to_fit();
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

class alignas(64) Chunk
{
public:

    [[nodiscard]] constexpr glm::vec3 world_origin() const
    {
        return glm::vec3 {
            static_cast<f32>(coord.x * CHUNK_SIZE),
            static_cast<f32>(coord.y * CHUNK_SIZE),
            static_cast<f32>(coord.z * CHUNK_SIZE),
        };
    }

    [[nodiscard]] static constexpr u32 get_index(u32 x, u32 y, u32 z) { return x + (y * CHUNK_SIZE) + (z * CHUNK_AREA); }

    [[nodiscard]] constexpr BlockID get_block_unchecked(u32 x, u32 y, u32 z) const { return m_blocks[get_index(x, y, z)]; }

    [[nodiscard]] constexpr BlockID get_block(u32 x, u32 y, u32 z) const
    {
        if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) return BlockType::Air;
        return m_blocks[get_index(x, y, z)];
    }

    constexpr void set_block_unchecked(u32 x, u32 y, u32 z, BlockID id) { m_blocks[get_index(x, y, z)] = id; }

    constexpr void set_block(u32 x, u32 y, u32 z, BlockID id)
    {
        if (x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE) return;

        const u32 index = get_index(x, y, z);
        const BlockID old_id = m_blocks[index];
        if (old_id == id) return;

        m_blocks[index] = id;

        const bool was_opaque = is_block_opaque(old_id);
        const bool is_opaque = is_block_opaque(id);

        if (was_opaque != is_opaque)
        {
            constexpr u32 last = CHUNK_SIZE - 1u;

            auto update_face = [&](Direction dir) {
                if (!is_opaque)
                {
                    // A solid block became transparent
                    m_portal_mask |= (1 << static_cast<u8>(dir));
                }
                else
                {
                    // A hole was filled, so the portal may be closed
                    recalculate_single_face(dir);
                }
            };

            if (x == 0) update_face(Direction::Left);
            if (x == last) update_face(Direction::Right);
            if (y == 0) update_face(Direction::Down);
            if (y == last) update_face(Direction::Up);
            if (z == 0) update_face(Direction::Back);
            if (z == last) update_face(Direction::Front);
        }
    }

    constexpr void recalculate_single_face(Direction dir)
    {
        m_portal_mask &= ~(1 << static_cast<u8>(dir));
        constexpr u32 last = CHUNK_SIZE - 1u;

        auto check_and_set = [&](u32 x, u32 y, u32 z) {
            if (!is_block_opaque(get_block_unchecked(x, y, z)))
            {
                m_portal_mask |= (1 << static_cast<u8>(dir));
                return true;
            }
            return false;
        };

        // Scan the affected face
        for (u32 a = 0; a < CHUNK_SIZE; ++a)
        {
            for (u32 b = 0; b < CHUNK_SIZE; ++b)
            {
                switch (dir)
                {
                    case Direction::Left:
                        if (check_and_set(0, a, b)) return;
                        break;
                    case Direction::Right:
                        if (check_and_set(last, a, b)) return;
                        break;
                    case Direction::Down:
                        if (check_and_set(a, 0, b)) return;
                        break;
                    case Direction::Up:
                        if (check_and_set(a, last, b)) return;
                        break;
                    case Direction::Back:
                        if (check_and_set(a, b, 0)) return;
                        break;
                    case Direction::Front:
                        if (check_and_set(a, b, last)) return;
                        break;
                }
            }
        }
    }

    [[nodiscard]] constexpr bool has_portal(Direction direction) const
    {
        return (m_portal_mask & (1 << static_cast<u8>(direction))) != 0;
    }

    void recalculate_portal_mask()
    {
        m_portal_mask = 0;
        constexpr u32 last = CHUNK_SIZE - 1u;

        auto scan_face = [this](Direction dir, auto get_coords) {
            for (u32 a = 0; a < CHUNK_SIZE; ++a)
            {
                for (u32 b = 0; b < CHUNK_SIZE; ++b)
                {
                    const auto [x, y, z] = get_coords(a, b);
                    if (!is_block_opaque(get_block_unchecked(x, y, z)))
                    {
                        m_portal_mask |= (1 << static_cast<u8>(dir));
                        return; // Found a hole
                    }
                }
            }
        };

        scan_face(Direction::Left, [](u32 y, u32 z) { return std::tuple {0u, y, z}; });
        scan_face(Direction::Right, [=](u32 y, u32 z) { return std::tuple {last, y, z}; });
        scan_face(Direction::Down, [](u32 x, u32 z) { return std::tuple {x, 0u, z}; });
        scan_face(Direction::Up, [=](u32 x, u32 z) { return std::tuple {x, last, z}; });
        scan_face(Direction::Back, [](u32 x, u32 y) { return std::tuple {x, y, 0u}; });
        scan_face(Direction::Front, [=](u32 x, u32 y) { return std::tuple {x, y, last}; });
    }

    constexpr void fill(BlockID id) { m_blocks.fill(id); }

    constexpr void clear() { m_blocks.fill(BlockType::Air); }

    [[nodiscard]] constexpr u32 count_non_air_blocks() const
    {
        u32 count = 0;
        for (const BlockID id : m_blocks)
            if (id != BlockType::Air) ++count;
        return count;
    }

    [[nodiscard]] BlockID sample_block(i32 x, i32 y, i32 z, const ChunkNeighbors& neighbors = {}) const;

    // Generates the visible surface mesh
    void generate_mesh(ChunkMesh& out_mesh, const ChunkNeighbors& neighbors = {}) const;

    // Chunk position in chunk-grid coordinates
    glm::ivec3 coord {0, 0, 0};

    bool mesh_dirty = false;

private:

    // 6 first bits: 1 if at least one non-opaque block for the corresponding face (L, R, D, U, B, F)
    u8 m_portal_mask = 0b11111100;

    // Flat array for contiguous memory
    std::array<BlockID, CHUNK_VOLUME> m_blocks {};
};

} // namespace vl