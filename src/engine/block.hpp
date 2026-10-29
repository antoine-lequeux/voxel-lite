#pragma once

#include "utils/types.hpp"

#include <array>
#include <glm/glm.hpp>
#include <string_view>
#include <utility>

namespace vl
{

using BlockID = u8;

enum class Direction : u8
{
    Left = 0,  // -X
    Right = 1, // +X
    Down = 2,  // -Y
    Up = 3,    // +Y
    Back = 4,  // -Z
    Front = 5, // +Z
};

inline constexpr usize DIRECTION_COUNT = 6;

[[nodiscard]] constexpr usize to_index(Direction dir)
{
    return std::to_underlying(dir);
}

[[nodiscard]] constexpr u8 direction_bit(Direction dir)
{
    return static_cast<u8>(1u << to_index(dir));
}

[[nodiscard]] constexpr Direction opposite(Direction dir)
{
    return static_cast<Direction>(to_index(dir) ^ 1u);
}

inline constexpr std::array<Direction, DIRECTION_COUNT> ALL_DIRECTIONS = {
    Direction::Left, Direction::Right, Direction::Down, Direction::Up, Direction::Back, Direction::Front,
};

inline constexpr std::array<glm::ivec3, DIRECTION_COUNT> DIRECTION_OFFSETS = {{
    {-1, 0, 0}, // Left (-X)
    {1, 0, 0},  // Right (+X)
    {0, -1, 0}, // Down (-Y)
    {0, 1, 0},  // Up (+Y)
    {0, 0, -1}, // Back (-Z)
    {0, 0, 1},  // Front (+Z)
}};

[[nodiscard]] constexpr glm::ivec3 to_offset(Direction dir)
{
    return DIRECTION_OFFSETS[to_index(dir)];
}

// Chunk portal connectivity encoding:
//   Bits 0..5  (6 bits)  -> 1 if the face has at least one non-opaque block
//   Bits 6..20 (15 bits) -> 1 if face pair {a, b} (a != b) is connected by non-opaque blocks inside the chunk
inline constexpr u32 ALL_OPEN_FACES_MASK = 0b00111111u;
inline constexpr u32 ALL_PORTAL_CONNECTIVITY = (1u << 21u) - 1u;

[[nodiscard]] consteval std::array<std::array<u32, DIRECTION_COUNT>, DIRECTION_COUNT> build_portal_pair_bits()
{
    std::array<std::array<u32, DIRECTION_COUNT>, DIRECTION_COUNT> table {};
    u32 next_pair_bit = 6u;
    for (usize a = 0; a < DIRECTION_COUNT; ++a)
    {
        table[a][a] = 1u << a;
        for (usize b = a + 1u; b < DIRECTION_COUNT; ++b)
        {
            const u32 bit = 1u << next_pair_bit++;
            table[a][b] = bit;
            table[b][a] = bit;
        }
    }
    return table;
}

inline constexpr auto PORTAL_PAIR_BITS = build_portal_pair_bits();

[[nodiscard]] constexpr u32 portal_pair_bit(Direction a, Direction b)
{
    return PORTAL_PAIR_BITS[to_index(a)][to_index(b)];
}

[[nodiscard]] constexpr u8 open_faces_mask(u32 portal_connectivity)
{
    return static_cast<u8>(portal_connectivity & ALL_OPEN_FACES_MASK);
}

// Returns the 6-bit mask of outgoing faces reachable when entering a chunk through this direction
[[nodiscard]] constexpr u8 reachable_faces_from(u32 portal_connectivity, Direction in_dir)
{
    u8 mask = 0u;
    for (Direction out_dir : ALL_DIRECTIONS)
        if ((portal_connectivity & portal_pair_bit(in_dir, out_dir)) != 0u) mask |= direction_bit(out_dir);
    return mask;
}

// Converts a 6-bit mask of faces touched by a single connected component into its 21-bit connectivity mask
[[nodiscard]] constexpr u32 component_faces_to_connectivity(u8 component_faces)
{
    u32 conn = component_faces;
    for (Direction a : ALL_DIRECTIONS)
    {
        if ((component_faces & direction_bit(a)) == 0u) continue;
        for (Direction b : ALL_DIRECTIONS)
            if ((component_faces & direction_bit(b)) != 0u) conn |= portal_pair_bit(a, b);
    }
    return conn;
}

namespace BlockType
{
inline constexpr BlockID Air = 0;
inline constexpr BlockID Grass = 1;
inline constexpr BlockID Dirt = 2;
inline constexpr BlockID Stone = 3;
inline constexpr BlockID Sand = 4;
inline constexpr BlockID Water = 5;
inline constexpr BlockID Snow = 6;
inline constexpr BlockID Wood = 7;
inline constexpr BlockID Leaves = 8;
inline constexpr u16 Count = 9;
} // namespace BlockType

enum class Visibility : u8
{
    Opaque = 0,
    Cutout = 1,
    Translucent = 2,
};

struct BlockDef
{
    std::string_view name = "Air";

    // Color per face indexed by direction
    std::array<glm::vec3, DIRECTION_COUNT> face_colors {};

    f32 roughness = 0.85f;

    bool is_solid = false;
    Visibility visibility = Visibility::Translucent;

    u8 light_opacity = 0;  // How much light is absorbed passing through (0 = clear, 15 = fully opaque)
    u8 light_emission = 0; // Self-emission level (0 = none, 15 = max brightness)

    [[nodiscard]] constexpr glm::vec3 color_for_face(Direction dir) const { return face_colors[to_index(dir)]; }

    [[nodiscard]] static constexpr std::array<glm::vec3, DIRECTION_COUNT> uniform_colors(glm::vec3 color)
    {
        return {color, color, color, color, color, color};
    }

    [[nodiscard]] static constexpr std::array<glm::vec3, DIRECTION_COUNT>
    top_bottom_side_colors(glm::vec3 top, glm::vec3 bottom, glm::vec3 side)
    {
        return {side, side, bottom, top, side, side};
    }
};

// Material entry uploaded once to a storage buffer
struct GpuBlockMaterial
{
    // xyz = face RGB color, w = surface roughness
    std::array<glm::vec4, DIRECTION_COUNT> face_color_and_roughness {};

    // x = normalized light emission [0..1], y = normalized opacity [0..1], z = visibility [0..2]
    // w is reserved
    glm::vec4 emission_and_opacity {};
};

static_assert(sizeof(GpuBlockMaterial) == 112, "GpuBlockMaterial must match 112-byte shader layout");

inline constexpr std::array<BlockDef, BlockType::Count> BLOCK_REGISTRY {{
    {
        .name = "Air",
        .face_colors = BlockDef::uniform_colors({0.0f, 0.0f, 0.0f}),
        .roughness = 0.0f,
        .is_solid = false,
        .visibility = Visibility::Translucent,
        .light_opacity = 0,
        .light_emission = 0,
    },
    {
        .name = "Grass",
        .face_colors = BlockDef::top_bottom_side_colors({0.34f, 0.74f, 0.25f}, {0.48f, 0.34f, 0.22f}, {0.42f, 0.56f, 0.26f}),
        .roughness = 0.85f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Dirt",
        .face_colors = BlockDef::uniform_colors({0.50f, 0.35f, 0.22f}),
        .roughness = 0.92f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Stone",
        .face_colors = BlockDef::uniform_colors({0.56f, 0.57f, 0.60f}),
        .roughness = 0.78f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Sand",
        .face_colors = BlockDef::uniform_colors({0.92f, 0.85f, 0.56f}),
        .roughness = 0.70f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Water",
        .face_colors = BlockDef::uniform_colors({0.18f, 0.52f, 0.86f}),
        .roughness = 0.12f,
        .is_solid = false,
        .visibility = Visibility::Translucent,
        .light_opacity = 2,
        .light_emission = 0,
    },
    {
        .name = "Snow",
        .face_colors = BlockDef::uniform_colors({0.94f, 0.96f, 0.99f}),
        .roughness = 0.60f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Wood",
        .face_colors = BlockDef::top_bottom_side_colors({0.65f, 0.50f, 0.31f}, {0.65f, 0.50f, 0.31f}, {0.38f, 0.26f, 0.15f}),
        .roughness = 0.88f,
        .is_solid = true,
        .visibility = Visibility::Opaque,
        .light_opacity = 15,
        .light_emission = 0,
    },
    {
        .name = "Leaves",
        .face_colors = BlockDef::uniform_colors({0.20f, 0.56f, 0.18f}),
        .roughness = 0.50f,
        .is_solid = true,
        .visibility = Visibility::Cutout,
        .light_opacity = 4,
        .light_emission = 0,
    },
}};

[[nodiscard]] constexpr const BlockDef& get_block_def(BlockID id)
{
    if (id >= BlockType::Count) return BLOCK_REGISTRY[BlockType::Air];
    return BLOCK_REGISTRY[id];
}

[[nodiscard]] constexpr bool is_block_opaque(BlockID id)
{
    return get_block_def(id).visibility == Visibility::Opaque;
}

[[nodiscard]] constexpr bool is_block_solid(BlockID id)
{
    return get_block_def(id).is_solid;
}

[[nodiscard]] constexpr bool should_render_face(BlockID current, BlockID neighbor)
{
    if (current == BlockType::Air) return false;

    const BlockDef& curr_def = get_block_def(current);
    const BlockDef& neigh_def = get_block_def(neighbor);

    // Nothing is rendered against a fully opaque neighbor
    if (neigh_def.visibility == Visibility::Opaque) return false;

    // Translucent volumes hide internal faces between identical blocks
    if (curr_def.visibility == Visibility::Translucent && current == neighbor) return false;

    // Everything else renders its face
    return true;
}

// Builds the material array from the block registry
[[nodiscard]] consteval std::array<GpuBlockMaterial, BlockType::Count> build_gpu_block_materials()
{
    std::array<GpuBlockMaterial, BlockType::Count> gpu_materials {};
    for (usize i = 0; i < BlockType::Count; ++i)
    {
        const BlockDef& def = BLOCK_REGISTRY[i];
        GpuBlockMaterial& mat = gpu_materials[i];
        for (usize face = 0; face < DIRECTION_COUNT; ++face)
        {
            const glm::vec3 c = def.face_colors[face];
            mat.face_color_and_roughness[face] = glm::vec4 {c.x, c.y, c.z, def.roughness};
        }
        mat.emission_and_opacity = glm::vec4 {
            static_cast<f32>(def.light_emission) / 15.0f,
            static_cast<f32>(def.light_opacity) / 15.0f,
            static_cast<f32>(std::to_underlying(def.visibility)),
            0.0f,
        };
    }
    return gpu_materials;
}

} // namespace vl
