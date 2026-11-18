#pragma once

#include "engine/block.hpp"
#include "terrain_config.hpp"

namespace vl
{

[[nodiscard]] constexpr i32 abs_i32(i32 v)
{
    return (v < 0) ? -v : v;
}

struct TreeInstance
{
    i32 root_x = 0;
    i32 root_y = 0;
    i32 root_z = 0;
    i32 trunk_height = world_config::TREE_MIN_TRUNK_HEIGHT;
};

// Fast deterministic 2D integer hash for procedural placement
[[nodiscard]] constexpr u32 hash_coords_2d(i32 x, i32 z, i32 seed)
{
    u32 h = static_cast<u32>(seed) ^ 0x9E3779B9u;
    h ^= static_cast<u32>(x) * 0x85EBCA6Bu;
    h = (h << 13u) | (h >> 19u);
    h ^= static_cast<u32>(z) * 0xC2B2AE35u;
    h *= 0x27D4EB2Fu;
    h ^= h >> 16u;
    return h;
}

// Floor division that rounds toward negative infinity for negative world coordinates
[[nodiscard]] constexpr i32 floor_div(i32 a, i32 b)
{
    const i32 q = a / b;
    const i32 r = a % b;
    return ((r != 0) && ((r < 0) != (b < 0))) ? (q - 1) : q;
}

// Determines whether world coordinate (wx, wz) is the chosen jittered tree root of its cell
[[nodiscard]] constexpr bool is_tree_cell_root(i32 wx, i32 wz, i32 seed)
{
    constexpr i32 CELL_SIZE = world_config::TREE_CELL_SIZE;
    const i32 cell_x = floor_div(wx, CELL_SIZE);
    const i32 cell_z = floor_div(wz, CELL_SIZE);

    const u32 h = hash_coords_2d(cell_x, cell_z, seed);
    if ((h % 100u) >= world_config::TREE_SPAWN_PROBABILITY_PERCENT) return false;

    // Align offsets to 2-block multiples so LOD 1 and LOD 2 sample identical tree roots
    constexpr u32 EVEN_POSITIONS = static_cast<u32>(CELL_SIZE / 2);
    const i32 offset_x = static_cast<i32>(((h >> 8u) % EVEN_POSITIONS) * 2u);
    const i32 offset_z = static_cast<i32>(((h >> 16u) % EVEN_POSITIONS) * 2u);

    const i32 root_wx = (cell_x * CELL_SIZE) + offset_x;
    const i32 root_wz = (cell_z * CELL_SIZE) + offset_z;

    return wx == root_wx && wz == root_wz;
}

// Checks all 2D surface conditions for spawning a tree at (wx, wz)
[[nodiscard]] constexpr bool
can_spawn_tree_at_column(i32 wx, i32 wz, f32 slope, f32 forest_noise, BlockID surface_block, i32 seed)
{
    if (surface_block != BlockType::Grass) return false;
    if (slope > world_config::TREE_MAX_SLOPE) return false;
    if (forest_noise < world_config::TREE_FOREST_THRESHOLD) return false;
    return is_tree_cell_root(wx, wz, seed);
}

// Deterministic trunk height in [TREE_MIN_TRUNK_HEIGHT, TREE_MAX_TRUNK_HEIGHT]
[[nodiscard]] constexpr i32 compute_tree_trunk_height(i32 wx, i32 wz, i32 seed)
{
    constexpr i32 RANGE = (world_config::TREE_MAX_TRUNK_HEIGHT - world_config::TREE_MIN_TRUNK_HEIGHT) + 1;
    const u32 h = hash_coords_2d(wx, wz, seed ^ 0x5A5A5A5A);
    return world_config::TREE_MIN_TRUNK_HEIGHT + static_cast<i32>(h % static_cast<u32>(RANGE));
}

// Evaluates whether voxel (wx, wy, wz) is occupied by 'tree', returning Wood, Leaves, or Air
[[nodiscard]] constexpr BlockID sample_tree_voxel(i32 wx, i32 wy, i32 wz, const TreeInstance& tree)
{
    const i32 trunk_top_y = tree.root_y + tree.trunk_height;
    const i32 canopy_min_y = trunk_top_y - world_config::TREE_CANOPY_BELOW_TOP;
    const i32 canopy_max_y = trunk_top_y + world_config::TREE_CANOPY_ABOVE_TOP;

    if (wy <= tree.root_y || wy > canopy_max_y) return BlockType::Air;

    const i32 dx = abs_i32(wx - tree.root_x);
    const i32 dz = abs_i32(wz - tree.root_z);

    // Trunk column
    if (dx == 0 && dz == 0 && wy <= trunk_top_y) return BlockType::Wood;

    // Leaf canopy
    if (wy < canopy_min_y) return BlockType::Air;

    const i32 dy = wy - trunk_top_y;
    constexpr i32 R = world_config::TREE_CANOPY_RADIUS;

    if (dy >= 2)
    {
        // Top tip of the crown (plus-shaped cross)
        if (dx + dz <= 1) return BlockType::Leaves;
        return BlockType::Air;
    }

    if (dy == 1)
    {
        // Upper crown layer (3x3 with rounded corners on shorter trees)
        if (dx <= 1 && dz <= 1)
        {
            if (dx == 1 && dz == 1 && (tree.trunk_height & 1) == 0) return BlockType::Air;
            return BlockType::Leaves;
        }
        return BlockType::Air;
    }

    // Main lower/middle canopy layers (5x5 with rounded corners)
    if (dx <= R && dz <= R)
    {
        if (dx == R && dz == R && dy != -1) return BlockType::Air;
        return BlockType::Leaves;
    }

    return BlockType::Air;
}

} // namespace vl
