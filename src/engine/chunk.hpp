#pragma once

#include "engine/block.hpp"
#include "renderer/vulkancontext.hpp"

namespace vl
{

inline constexpr i32 CHUNK_SIZE = 32;
inline constexpr i32 CHUNK_AREA = CHUNK_SIZE * CHUNK_SIZE;
inline constexpr i32 CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

inline constexpr u32 MAX_LOD = 5;

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
    return (x & 0x3Fu) | ((y & 0x3Fu) << 6u) | ((z & 0x3Fu) << 12u) | ((static_cast<u32>(to_index(normal)) & 0x07u) << 18u) |
           ((ao_level & 0x03u) << 21u) | ((uv_id & 0x03u) << 23u) | ((static_cast<u32>(block_id) & 0x7Fu) << 25u);
}

inline constexpr u32 INVALID_DRAW_SLOT = std::numeric_limits<u32>::max();

// Per-chunk draw metadata read on the GPU by shaders
struct alignas(16) GpuChunkDraw
{
    glm::ivec4 origin_and_quads {0, 0, 0, 0}; // xyz = chunk world origin in blocks, w = quad_count (0 = empty)
    u64 vertex_buffer_address = 0;
    u64 reserved = 0;
};

static_assert(sizeof(GpuChunkDraw) == 32);

// Persistent CPU-side table of active chunk meshes mirrored incrementally to a GPU buffer for indirect rendering
class ChunkDrawTable
{
public:

    static ChunkDrawTable& instance();

    u32 update_slot(u32 current_slot, const glm::ivec3& coord, u32 quad_count, u64 device_address);
    void free_slot(u32 slot);
    void reset();

    [[nodiscard]] u32 active_count() const { return m_active_count; }
    [[nodiscard]] u32 slot_limit() const { return static_cast<u32>(m_entries.size()); }
    [[nodiscard]] bool has_dirty_slots() const { return !m_dirty_slots.empty(); }

    void flush_dirty(vulk::VulkanContext& ctx, vulk::BufferHandle gpu_buffer);

private:

    void mark_dirty(u32 slot);

    std::vector<GpuChunkDraw> m_entries {};
    std::vector<u32> m_free_slots {};
    std::vector<u32> m_dirty_slots {};
    std::vector<u8> m_dirty_flags {};
    u32 m_active_count = 0;
};

struct ChunkMesh
{
    glm::ivec3 coord {0, 0, 0};
    std::vector<PackedVertex> vertices {};
    u32 quad_count = 0;
    u32 draw_slot = INVALID_DRAW_SLOT;
    bool meshed_once = false;

    // GPU storage buffer for vertex pulling
    vulk::BufferHandle vertex_buffer {};
    u64 device_address = 0;
    u64 buffer_capacity_bytes = 0;

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
        quad_count = 0;
    }

    // Transfers ownership of an existing GPU vertex buffer so a chunk changing LOD continues rendering
    // its previous mesh until its replacement mesh is uploaded
    void inherit_gpu_resources(ChunkMesh& donor)
    {
        vertex_buffer = donor.vertex_buffer;
        device_address = donor.device_address;
        buffer_capacity_bytes = donor.buffer_capacity_bytes;
        quad_count = donor.quad_count;
        draw_slot = donor.draw_slot;
        meshed_once = donor.meshed_once;
        donor.vertex_buffer = {};
        donor.device_address = 0;
        donor.buffer_capacity_bytes = 0;
        donor.quad_count = 0;
        donor.draw_slot = INVALID_DRAW_SLOT;
    }

    // Uploads the generated PackedVertex array to VRAM and ensures the shared quad index buffer is large enough
    bool upload_to_gpu(vulk::VulkanContext& ctx);
    void destroy_gpu_resources(vulk::VulkanContext& ctx);
};

template <u32 Lod = 1>
    requires(Lod >= 1 && Lod <= MAX_LOD)
class Chunk;

// View of an adjacent chunk for cross-chunk face culling
struct NeighborChunk
{
    enum class Kind : u8
    {
        Air = 0,   // Air chunk
        Solid = 1, // Ungenerated chunk (treated as opaque)
        Real = 2,  // Loaded chunk
    };

    const void* chunk_ptr = nullptr;
    Kind kind = Kind::Air;
    u8 lod = 1;

    [[nodiscard]] static constexpr NeighborChunk air() { return {}; }
    [[nodiscard]] static constexpr NeighborChunk solid() { return {.chunk_ptr = nullptr, .kind = Kind::Solid, .lod = 1}; }

    template <u32 Lod>
    [[nodiscard]] static constexpr NeighborChunk from_chunk(const Chunk<Lod>& chunk)
    {
        return {.chunk_ptr = &chunk, .kind = Kind::Real, .lod = static_cast<u8>(Lod)};
    }

    [[nodiscard]] BlockID sample_meter(u32 x_m, u32 y_m, u32 z_m) const;

    [[nodiscard]] bool
    should_render_boundary_face(BlockID current, u8 self_lod, u32 x_m, u32 y_m, u32 z_m, u32 sx, u32 sy, u32 sz) const;
};

// Neighbor references for cross-chunk face culling
struct ChunkNeighbors
{
    NeighborChunk left {};  // -X
    NeighborChunk right {}; // +X
    NeighborChunk down {};  // -Y
    NeighborChunk up {};    // +Y
    NeighborChunk back {};  // -Z
    NeighborChunk front {}; // +Z
};

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
class Chunk
{
public:

    static constexpr u32 LOD = Lod;
    static constexpr u32 SCALE = 1u << (Lod - 1u);                         // Size of each macroblock in meters (1, 2, 4, 8, 16)
    static constexpr u32 DIM = static_cast<u32>(CHUNK_SIZE) >> (Lod - 1u); // Macroblocks per axis (32, 16, 8, 4, 2)
    static constexpr u32 AREA = DIM * DIM;
    static constexpr u32 VOLUME = DIM * DIM * DIM;

    [[nodiscard]] constexpr glm::vec3 world_origin() const
    {
        return glm::vec3 {
            static_cast<f32>(coord.x * CHUNK_SIZE),
            static_cast<f32>(coord.y * CHUNK_SIZE),
            static_cast<f32>(coord.z * CHUNK_SIZE),
        };
    }

    [[nodiscard]] static constexpr u32 get_index(u32 x, u32 y, u32 z) { return x + (y * DIM) + (z * AREA); }

    [[nodiscard]] constexpr BlockID get_block_unchecked(u32 x, u32 y, u32 z) const { return m_blocks[get_index(x, y, z)]; }

    [[nodiscard]] constexpr BlockID get_block(u32 x, u32 y, u32 z) const
    {
        if (x >= DIM || y >= DIM || z >= DIM) return BlockType::Air;
        return m_blocks[get_index(x, y, z)];
    }

    [[nodiscard]] constexpr BlockID get_block_at_meter(u32 x_m, u32 y_m, u32 z_m) const
    {
        constexpr u32 shift = Lod - 1u;
        return get_block_unchecked(x_m >> shift, y_m >> shift, z_m >> shift);
    }

    constexpr void set_block_unchecked(u32 x, u32 y, u32 z, BlockID id) { m_blocks[get_index(x, y, z)] = id; }

    constexpr void set_block(u32 x, u32 y, u32 z, BlockID id)
    {
        if (x >= DIM || y >= DIM || z >= DIM) return;

        const u32 index = get_index(x, y, z);
        const BlockID old_id = m_blocks[index];
        if (old_id == id) return;

        m_blocks[index] = id;

        const bool was_opaque = is_block_opaque(old_id);
        const bool is_opaque = is_block_opaque(id);

        if (was_opaque != is_opaque)
        {
            constexpr u32 last = DIM - 1u;

            auto update_face = [&](Direction dir) {
                if (!is_opaque)
                    m_portal_mask |= direction_bit(dir);
                else
                    recalculate_single_face(dir);
            };

            if (x == 0u) update_face(Direction::Left);
            if (x == last) update_face(Direction::Right);
            if (y == 0u) update_face(Direction::Down);
            if (y == last) update_face(Direction::Up);
            if (z == 0u) update_face(Direction::Back);
            if (z == last) update_face(Direction::Front);
        }
    }

    constexpr void recalculate_single_face(Direction dir)
    {
        m_portal_mask &= static_cast<u8>(~direction_bit(dir));
        constexpr u32 last = DIM - 1u;

        auto check_and_set = [&](u32 x, u32 y, u32 z) {
            if (!is_block_opaque(get_block_unchecked(x, y, z)))
            {
                m_portal_mask |= static_cast<u8>(1u << static_cast<u8>(dir));
                return true;
            }
            return false;
        };

        for (u32 a = 0; a < DIM; ++a)
        {
            for (u32 b = 0; b < DIM; ++b)
            {
                switch (dir)
                {
                    case Direction::Left:
                        if (check_and_set(0u, a, b)) return;
                        break;
                    case Direction::Right:
                        if (check_and_set(last, a, b)) return;
                        break;
                    case Direction::Down:
                        if (check_and_set(a, 0u, b)) return;
                        break;
                    case Direction::Up:
                        if (check_and_set(a, last, b)) return;
                        break;
                    case Direction::Back:
                        if (check_and_set(a, b, 0u)) return;
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
        return (m_portal_mask & direction_bit(direction)) != 0u;
    }

    constexpr void recalculate_portal_mask()
    {
        m_portal_mask = 0;
        constexpr u32 last = DIM - 1u;

        auto scan_face = [this](Direction dir, auto get_coords) {
            for (u32 a = 0; a < DIM; ++a)
            {
                for (u32 b = 0; b < DIM; ++b)
                {
                    const auto [x, y, z] = get_coords(a, b);
                    if (!is_block_opaque(get_block_unchecked(x, y, z)))
                    {
                        m_portal_mask |= static_cast<u8>(1u << static_cast<u8>(dir));
                        return;
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

    // Generates the visible surface mesh scaled to match the physical chunk size
    void generate_mesh(ChunkMesh& out_mesh, const ChunkNeighbors& neighbors = {}) const;

    // Chunk position in chunk-grid coordinates
    glm::ivec3 coord {0, 0, 0};

    bool mesh_dirty = false;

private:

    // 6 first bits: 1 if at least one non-opaque block for the corresponding face (L, R, D, U, B, F)
    u8 m_portal_mask = 0b00111111;

    std::array<BlockID, VOLUME> m_blocks {};
};

} // namespace vl
