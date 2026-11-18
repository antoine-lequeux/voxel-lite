#pragma once

#include "engine/block.hpp"
#include "heightmap.hpp"
#include "terrain_config.hpp"

namespace vl
{

// Returns true if the 3D cave noises carve this underground voxel into Air
[[nodiscard]] constexpr bool is_cave_carved(f32 spaghetti_noise, f32 large_cave_noise, i32 col_height, i32 depth_below_surface)
{
    // Never carve cave holes through underwater seabeds or low shorelines
    if (col_height <= world_config::SEA_LEVEL + 3 && depth_below_surface < world_config::CAVE_UNDERWATER_ROOF_DEPTH) return false;

    // Large caves only form well below the surface crust
    if (depth_below_surface >= world_config::CAVE_LARGE_MIN_DEPTH && large_cave_noise > world_config::LARGE_CAVE_THRESHOLD)
        return true;

    // Near the surface crust, only allow rare spaghetti cave entrances where the second noise also exceeds a threshold
    if (depth_below_surface < world_config::CAVE_SURFACE_ROOF_DEPTH &&
        large_cave_noise < world_config::CAVE_ENTRANCE_RARITY_THRESHOLD)
    {
        return false;
    }

    return abs_f32(spaghetti_noise - world_config::SPAGHETTI_CAVE_CENTER) < world_config::SPAGHETTI_CAVE_HALF_WIDTH;
}

// Determines the top surface block of a column based on elevation, slope, and surface patch noise
[[nodiscard]] constexpr BlockID choose_surface_block(i32 col_height, f32 slope, f32 patch_noise)
{
    // Steep cliffs and steep mountain faces expose bare stone
    if (slope >= world_config::CLIFF_SLOPE_THRESHOLD) return BlockType::Stone;
    if (col_height >= world_config::HIGH_PEAK_STONE_HEIGHT && slope >= world_config::HIGH_PEAK_SLOPE_THRESHOLD)
        return BlockType::Stone;

    // Underwater floor (below y = 0): clay and sand
    if (col_height < world_config::SEA_LEVEL)
    {
        if (col_height < world_config::UNDERWATER_DEEP_CLAY_HEIGHT)
            return (patch_noise < 0.72f) ? BlockType::Clay : BlockType::Sand;

        return (patch_noise < world_config::UNDERWATER_CLAY_NOISE_THRESHOLD) ? BlockType::Clay : BlockType::Sand;
    }

    // Coastal beaches next to water: sand in some areas, grass in others
    if (col_height <= world_config::BEACH_MAX_HEIGHT)
    {
        if (patch_noise > world_config::BEACH_SAND_NOISE_THRESHOLD) return BlockType::Sand;
    }

    // High peaks: snow above the snowline
    const f32 snow_threshold =
        static_cast<f32>(world_config::SNOW_BASE_HEIGHT) + ((patch_noise - 0.5f) * world_config::SNOW_NOISE_AMPLITUDE);
    if (static_cast<f32>(col_height) >= snow_threshold) return BlockType::Snow;

    // Default grass block everywhere else
    return BlockType::Grass;
}

// Chooses the solid block at world height 'wy_base' (for wy_base <= col_height)
[[nodiscard]] constexpr BlockID
paint_solid_block(i32 wy_base, i32 scale, i32 col_height, BlockID surface_block, bool has_solid_block_above)
{
    const i32 depth = col_height - wy_base;

    if (surface_block == BlockType::Stone) return BlockType::Stone;

    if (surface_block == BlockType::Grass)
    {
        if (depth > world_config::DIRT_DEPTH + scale) return BlockType::Stone;
        return has_solid_block_above ? BlockType::Dirt : BlockType::Grass;
    }

    if (surface_block == BlockType::Sand) return (depth <= world_config::SAND_DEPTH + scale) ? BlockType::Sand : BlockType::Stone;

    if (surface_block == BlockType::Clay)
        return (depth <= world_config::UNDERWATER_SEDIMENT_DEPTH + scale) ? BlockType::Clay : BlockType::Stone;

    if (surface_block == BlockType::Snow) return (depth <= world_config::SNOW_DEPTH + scale) ? BlockType::Snow : BlockType::Stone;

    return BlockType::Stone;
}

} // namespace vl
