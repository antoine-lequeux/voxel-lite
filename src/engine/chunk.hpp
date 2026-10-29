#pragma once

#include "block.hpp"
#include "renderer/vulkancontext.hpp"

#include <bit>
#include <variant>

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

    constexpr bool operator==(const GpuChunkDraw& other) const
    {
        return origin_and_quads == other.origin_and_quads && vertex_buffer_address == other.vertex_buffer_address &&
               reserved == other.reserved;
    }
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

// Representation of adjacent chunk states for cross-chunk face culling
struct AirNeighbor
{
};
struct SolidNeighbor
{
};

using NeighborTarget =
    std::variant<AirNeighbor, SolidNeighbor, const Chunk<1>*, const Chunk<2>*, const Chunk<3>*, const Chunk<4>*, const Chunk<5>*>;

struct NeighborChunk
{
    NeighborTarget target {AirNeighbor {}};
    u8 lod = 1;

    [[nodiscard]] static constexpr NeighborChunk air() { return NeighborChunk {AirNeighbor {}, 1}; }
    [[nodiscard]] static constexpr NeighborChunk solid() { return NeighborChunk {SolidNeighbor {}, 1}; }

    template <u32 Lod>
    [[nodiscard]] static constexpr NeighborChunk from_chunk(const Chunk<Lod>& chunk)
    {
        return NeighborChunk {&chunk, static_cast<u8>(Lod)};
    }

    [[nodiscard]] constexpr bool is_air() const { return std::holds_alternative<AirNeighbor>(target); }
    [[nodiscard]] constexpr bool is_solid() const { return std::holds_alternative<SolidNeighbor>(target); }

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

    [[nodiscard]] const NeighborChunk& get(Direction dir) const
    {
        switch (dir)
        {
            case Direction::Left: return left;
            case Direction::Right: return right;
            case Direction::Down: return down;
            case Direction::Up: return up;
            case Direction::Back: return back;
            case Direction::Front: return front;
        }
        return left;
    }
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

        if (was_opaque != is_opaque) recalculate_portal_mask(true);
    }

    [[nodiscard]] constexpr bool face_has_portal(Direction dir) const
    {
        constexpr u32 last = DIM - 1u;

        auto has_transparent_block = [&]<typename CoordFn>(CoordFn get_coords) {
            for (u32 a = 0; a < DIM; ++a)
            {
                for (u32 b = 0; b < DIM; ++b)
                {
                    const auto [x, y, z] = get_coords(a, b);
                    if (!is_block_opaque(get_block_unchecked(x, y, z))) return true;
                }
            }
            return false;
        };

        switch (dir)
        {
            case Direction::Left: return has_transparent_block([](u32 y, u32 z) { return std::tuple {0u, y, z}; });
            case Direction::Right: return has_transparent_block([=](u32 y, u32 z) { return std::tuple {last, y, z}; });
            case Direction::Down: return has_transparent_block([](u32 x, u32 z) { return std::tuple {x, 0u, z}; });
            case Direction::Up: return has_transparent_block([=](u32 x, u32 z) { return std::tuple {x, last, z}; });
            case Direction::Back: return has_transparent_block([](u32 x, u32 y) { return std::tuple {x, y, 0u}; });
            case Direction::Front: return has_transparent_block([=](u32 x, u32 y) { return std::tuple {x, y, last}; });
        }
        return false;
    }

    [[nodiscard]] constexpr bool has_portal(Direction direction) const
    {
        return (m_portal_mask & direction_bit(direction)) != 0u;
    }

    [[nodiscard]] constexpr u32 portal_connectivity() const { return m_portal_mask; }

    constexpr void recalculate_portal_mask(bool full_3d_connectivity = true)
    {
        u8 open_faces = 0u;
        for (Direction dir : ALL_DIRECTIONS)
            if (face_has_portal(dir)) open_faces |= direction_bit(dir);

        if (open_faces == 0u)
        {
            m_portal_mask = 0u;
            return;
        }

        if (std::popcount(open_faces) <= 1)
        {
            m_portal_mask = open_faces;
            return;
        }

        const u32 max_connectivity = component_faces_to_connectivity(open_faces);
        if (!full_3d_connectivity)
        {
            m_portal_mask = max_connectivity;
            return;
        }

        constexpr u32 last = DIM - 1u;
        constexpr u32 shift = static_cast<u32>(std::countr_zero(DIM));
        constexpr u32 coord_mask = DIM - 1u;

        std::array<u64, (VOLUME + 63u) / 64u> visited {};
        std::array<u16, VOLUME> queue {};
        u32 connectivity = open_faces;

        auto try_visit = [&](u32 idx) -> bool {
            const u32 word = idx >> 6u;
            const u64 bit = 1ull << (idx & 63u);
            if ((visited[word] & bit) != 0ull) return false;
            visited[word] |= bit;
            return true;
        };

        auto flood_from_seed = [&](u32 seed_idx) {
            if (is_block_opaque(m_blocks[seed_idx]) || !try_visit(seed_idx)) return;

            u8 comp_faces = 0u;
            u32 head = 0u;
            u32 tail = 0u;
            queue[tail++] = static_cast<u16>(seed_idx);

            while (head < tail)
            {
                const u32 cur = queue[head++];
                const u32 x = cur & coord_mask;
                const u32 y = (cur >> shift) & coord_mask;
                const u32 z = cur >> (2u * shift);

                if (x == 0u) comp_faces |= direction_bit(Direction::Left);
                if (x == last) comp_faces |= direction_bit(Direction::Right);
                if (y == 0u) comp_faces |= direction_bit(Direction::Down);
                if (y == last) comp_faces |= direction_bit(Direction::Up);
                if (z == 0u) comp_faces |= direction_bit(Direction::Back);
                if (z == last) comp_faces |= direction_bit(Direction::Front);

                if (x > 0u)
                {
                    const u32 n = cur - 1u;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
                if (x < last)
                {
                    const u32 n = cur + 1u;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
                if (y > 0u)
                {
                    const u32 n = cur - DIM;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
                if (y < last)
                {
                    const u32 n = cur + DIM;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
                if (z > 0u)
                {
                    const u32 n = cur - AREA;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
                if (z < last)
                {
                    const u32 n = cur + AREA;
                    if (!is_block_opaque(m_blocks[n]) && try_visit(n)) queue[tail++] = static_cast<u16>(n);
                }
            }

            connectivity |= component_faces_to_connectivity(comp_faces);
        };

        auto seed_face = [&]<typename CoordFn>(Direction dir, CoordFn get_coords) {
            if ((open_faces & direction_bit(dir)) == 0u || connectivity == max_connectivity) return;
            for (u32 a = 0; a < DIM && connectivity != max_connectivity; ++a)
            {
                for (u32 b = 0; b < DIM && connectivity != max_connectivity; ++b)
                {
                    const auto [x, y, z] = get_coords(a, b);
                    flood_from_seed(get_index(x, y, z));
                }
            }
        };

        seed_face(Direction::Left, [](u32 y, u32 z) { return std::tuple {0u, y, z}; });
        seed_face(Direction::Right, [=](u32 y, u32 z) { return std::tuple {last, y, z}; });
        seed_face(Direction::Down, [](u32 x, u32 z) { return std::tuple {x, 0u, z}; });
        seed_face(Direction::Up, [=](u32 x, u32 z) { return std::tuple {x, last, z}; });
        seed_face(Direction::Back, [](u32 x, u32 y) { return std::tuple {x, y, 0u}; });
        seed_face(Direction::Front, [=](u32 x, u32 y) { return std::tuple {x, y, last}; });

        m_portal_mask = connectivity;
    }

    constexpr void fill(BlockID id)
    {
        m_blocks.fill(id);
        m_portal_mask = is_block_opaque(id) ? 0u : ALL_PORTAL_CONNECTIVITY;
    }

    constexpr void clear()
    {
        m_blocks.fill(BlockType::Air);
        m_portal_mask = ALL_PORTAL_CONNECTIVITY;
    }

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

    // 21-bit face and face-pair portal connectivity mask:
    //   Bits 0..5   -> open faces (L, R, D, U, B, F)
    //   Bits 6..20  -> connected face pairs through non-opaque interior paths
    u32 m_portal_mask = ALL_PORTAL_CONNECTIVITY;

    std::array<BlockID, VOLUME> m_blocks {};
};

} // namespace vl
