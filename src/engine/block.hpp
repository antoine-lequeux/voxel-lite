#pragma once

#include "utils/types.hpp"

#include <array>
#include <glm/glm.hpp>
#include <string_view>

namespace vl
{

using BlockID = u16;

enum class Direction : u8
{
    Left = 0,  // -X
    Right = 1, // +X
    Down = 2,  // -Y
    Up = 3,    // +Y
    Back = 4,  // -Z
    Front = 5, // +Z
};

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
    std::array<glm::vec3, 6> face_colors {};

    f32 roughness = 0.85f;

    bool is_solid = false;
    Visibility visibility = Visibility::Translucent;

    u8 light_opacity = 0;  // How much light is absorbed passing through (0 = clear, 15 = fully opaque)
    u8 light_emission = 0; // Self-emission level (0 = none, 15 = max brightness)

    [[nodiscard]] constexpr glm::vec3 color_for_face(Direction dir) const { return face_colors[static_cast<usize>(dir)]; }

    [[nodiscard]] static constexpr std::array<glm::vec3, 6> uniform_colors(glm::vec3 color)
    {
        return {color, color, color, color, color, color};
    }

    [[nodiscard]] static constexpr std::array<glm::vec3, 6>
    top_bottom_side_colors(glm::vec3 top, glm::vec3 bottom, glm::vec3 side)
    {
        return {side, side, bottom, top, side, side};
    }
};

// Material entry uploaded once to a storage buffer
struct GpuBlockMaterial
{
    // xyz = face RGB color, w = surface roughness
    std::array<glm::vec4, 6> face_color_and_roughness {};

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
        for (usize face = 0; face < 6; ++face)
        {
            const glm::vec3 c = def.face_colors[face];
            mat.face_color_and_roughness[face] = glm::vec4 {c.x, c.y, c.z, def.roughness};
        }
        mat.emission_and_opacity = glm::vec4 {
            static_cast<f32>(def.light_emission) / 15.0f,
            static_cast<f32>(def.light_opacity) / 15.0f,
            static_cast<f32>(def.visibility),
            0.0f,
        };
    }
    return gpu_materials;
}

} // namespace vl
