#pragma once

#include "utils/types.hpp"

#include <array>

namespace vl::world_config
{

// ------------------------------------------------------
// --- Global world elevations and detail thresholds: ---
// ------------------------------------------------------

inline constexpr i32 SEA_LEVEL = 0;

// Maximum LOD level (inclusive) at which 3D caves and trees are evaluated
inline constexpr u32 MAX_CAVE_LOD = 1;
inline constexpr u32 MAX_TREE_LOD = 2;

// Extra border columns sampled on each side of a chunk for slope and tree canopy overhang
inline constexpr i32 COLUMN_PADDING = 3;

// ----------------------------------
// --- Continentality noise (2D): ---
// ----------------------------------

inline constexpr i32 CONT_SEED_OFFSET = 101;
inline constexpr f32 CONT_FEATURE_SCALE = 1800.0f;
inline constexpr i32 CONT_OCTAVES = 3;
inline constexpr f32 CONT_GAIN = 0.45f;
inline constexpr f32 CONT_LACUNARITY = 0.5f;
inline constexpr f32 CONT_WARP_SCALE = 800.0f;
inline constexpr f32 CONT_WARP_AMPLITUDE = 140.0f;
inline constexpr f32 CONT_RAW_MIN = -1.15f;
inline constexpr f32 CONT_RAW_MAX = 1.15f;

// ---------------------------------------------------------
// --- Continentality base height spline control points: ---
// ---------------------------------------------------------

struct HeightControlPoint
{
    f32 continentality = 0.0f;
    f32 height = 0.0f;
};

inline constexpr f32 CONT_VAL_DEEP_OCEAN = 0.00f;
inline constexpr f32 CONT_HEIGHT_DEEP_OCEAN = -100.0f;

inline constexpr f32 CONT_VAL_OCEAN_FLOOR_END = 0.28f;
inline constexpr f32 CONT_HEIGHT_OCEAN_FLOOR_END = -75.0f;

inline constexpr f32 CONT_VAL_OCEAN_CLIFF_END = 0.38f;
inline constexpr f32 CONT_HEIGHT_OCEAN_CLIFF_END = -16.0f;

inline constexpr f32 CONT_VAL_BEACH_FLOOR_END = 0.46f;
inline constexpr f32 CONT_HEIGHT_BEACH_FLOOR_END = -10.0f;

inline constexpr f32 CONT_VAL_BEACH_END = 0.54f;
inline constexpr f32 CONT_HEIGHT_BEACH_END = 8.0f;

inline constexpr f32 CONT_VAL_PLAINS_END = 0.76f;
inline constexpr f32 CONT_HEIGHT_PLAINS_END = 28.0f;

inline constexpr f32 CONT_VAL_MAX_INLAND = 1.00f;
inline constexpr f32 CONT_HEIGHT_MAX_INLAND = 150.0f;

inline constexpr std::array<HeightControlPoint, 7> CONTINENTALITY_SPLINE {{
    {CONT_VAL_DEEP_OCEAN, CONT_HEIGHT_DEEP_OCEAN},
    {CONT_VAL_OCEAN_FLOOR_END, CONT_HEIGHT_OCEAN_FLOOR_END},
    {CONT_VAL_OCEAN_CLIFF_END, CONT_HEIGHT_OCEAN_CLIFF_END},
    {CONT_VAL_BEACH_FLOOR_END, CONT_HEIGHT_BEACH_FLOOR_END},
    {CONT_VAL_BEACH_END, CONT_HEIGHT_BEACH_END},
    {CONT_VAL_PLAINS_END, CONT_HEIGHT_PLAINS_END},
    {CONT_VAL_MAX_INLAND, CONT_HEIGHT_MAX_INLAND},
}};

// ---------------------------
// --- Erosion noise (2D): ---
// ---------------------------

inline constexpr i32 EROSION_SEED_OFFSET = 202;
inline constexpr f32 EROSION_FEATURE_SCALE = 420.0f;
inline constexpr i32 EROSION_OCTAVES = 6;
inline constexpr f32 EROSION_GAIN = 0.55f;
inline constexpr f32 EROSION_LACUNARITY = 2.0f;
inline constexpr f32 EROSION_WEIGHTED_STRENGTH = 0.35f;
inline constexpr f32 EROSION_WARP_SCALE = 170.0f;
inline constexpr f32 EROSION_WARP_AMPLITUDE = 45.0f;
inline constexpr f32 EROSION_RAW_MIN = -0.95f;
inline constexpr f32 EROSION_RAW_MAX = 1.10f;

inline constexpr f32 EROSION_CURVE_POWER = 1.35f;
inline constexpr f32 EROSION_MAX_HEIGHT_REDUCTION = 80.0f;

// -------------------------
// --- Peaks noise (2D): ---
// -------------------------

inline constexpr i32 PEAKS_ISOLATED_SEED_OFFSET = 303;
inline constexpr i32 PEAKS_RANGE_SEED_OFFSET = 307;
inline constexpr i32 PEAKS_MASK_SEED_OFFSET = 311;
inline constexpr i32 PEAKS_SHAPE_SEED_OFFSET = 313;
inline constexpr i32 PEAKS_WARP_SEED_OFFSET = 317;

inline constexpr f32 PEAKS_ISOLATED_CELL_SCALE = 360.0f;
inline constexpr f32 PEAKS_RANGE_CELL_SCALE = 190.0f;
inline constexpr f32 PEAKS_RANGE_MASK_SCALE = 650.0f;
inline constexpr i32 PEAKS_RANGE_MASK_OCTAVES = 3;
inline constexpr f32 PEAKS_WARP_SCALE = 150.0f;
inline constexpr f32 PEAKS_WARP_AMPLITUDE = 48.0f;
inline constexpr f32 PEAKS_GRID_JITTER = 1.0f;
inline constexpr f32 PEAKS_SIZE_JITTER = 0.35f;
inline constexpr f32 PEAKS_CIRCLE_RADIUS = 0.20f;
inline constexpr f32 PEAKS_BLEND_SMOOTHNESS = 0.12f;

// Shaping between pointy peaks and flat-topped peaks
inline constexpr f32 PEAKS_POINTY_EXPONENT = 1.85f;
inline constexpr f32 PEAKS_FLAT_PLATEAU_CAP = 0.68f;
inline constexpr f32 PEAKS_FLAT_SMOOTHNESS = 0.12f;
inline constexpr f32 PEAKS_SHAPE_SELECTOR_SCALE = 520.0f;

inline constexpr f32 PEAKS_CURVE_POWER = 1.15f;
inline constexpr f32 PEAKS_MAX_HEIGHT_BONUS = 115.0f;

// -----------------------
// --- 3D cave noises: ---
// -----------------------

inline constexpr i32 CAVE_SPAGHETTI_SEED_OFFSET = 1009;
inline constexpr i32 CAVE_LARGE_SEED_OFFSET = 2027;
inline constexpr i32 CAVE_SECOND_SHEET_SEED_OFFSET = 503;
inline constexpr i32 CAVE_REGION_SEED_OFFSET = 809;

// 3D ridged sheets whose intersection forms smooth cylindrical 3D cave tubes
inline constexpr f32 CAVE_FEATURE_SCALE = 115.0f;
inline constexpr i32 CAVE_OCTAVES = 2;
inline constexpr f32 CAVE_GAIN = 0.35f;
inline constexpr f32 CAVE_LACUNARITY = 2.0f;
inline constexpr f32 CAVE_WEIGHTED_STRENGTH = 0.50f;

// Slow 3D regional modulator so cave systems form localized networks rather than everywhere
inline constexpr f32 CAVE_REGION_SCALE = 250.0f;
inline constexpr f32 CAVE_REGION_BIAS_MIN = -0.22f;
inline constexpr f32 CAVE_REGION_BIAS_MAX = 0.04f;

// Piecewise plateau remap
inline constexpr f32 CAVE_TUBE_RAW_MIN = 0.58f;
inline constexpr f32 CAVE_TUBE_RAW_MAX = 0.94f;
inline constexpr f32 CAVE_CHAMBER_RAW_MIN = 1.14f;
inline constexpr f32 CAVE_CHAMBER_RAW_MAX = 1.30f;

// Spaghetti caves carve blocks where noise 1 is close to 0.5
inline constexpr f32 SPAGHETTI_CAVE_CENTER = 0.50f;
inline constexpr f32 SPAGHETTI_CAVE_HALF_WIDTH = 0.045f;

// Large caves carve blocks where noise 2 exceeds 0.8
inline constexpr f32 LARGE_CAVE_THRESHOLD = 0.80f;

// Surface crust protection to keep surface cave entrances rare and prevent seabed holes
inline constexpr i32 CAVE_SURFACE_ROOF_DEPTH = 31;
inline constexpr i32 CAVE_LARGE_MIN_DEPTH = 32;
inline constexpr i32 CAVE_UNDERWATER_ROOF_DEPTH = 20;
inline constexpr f32 CAVE_ENTRANCE_RARITY_THRESHOLD = 0.44f;

// ------------------------------------------------------
// --- Block painting rules (surface and subsurface): ---
// ------------------------------------------------------

inline constexpr i32 SURFACE_PATCH_SEED_OFFSET = 401;
inline constexpr f32 SURFACE_PATCH_SCALE = 96.0f;
inline constexpr i32 SURFACE_PATCH_OCTAVES = 2;

inline constexpr i32 DIRT_DEPTH = 4;
inline constexpr i32 SAND_DEPTH = 4;
inline constexpr i32 UNDERWATER_SEDIMENT_DEPTH = 3;
inline constexpr i32 SNOW_DEPTH = 2;

// Underwater (col_height < SEA_LEVEL): clay and sand
inline constexpr i32 UNDERWATER_DEEP_CLAY_HEIGHT = -26;
inline constexpr f32 UNDERWATER_CLAY_NOISE_THRESHOLD = 0.46f;

// Next to water (SEA_LEVEL <= col_height <= BEACH_MAX_HEIGHT): sand in some places
inline constexpr i32 BEACH_MAX_HEIGHT = 7;
inline constexpr f32 BEACH_SAND_NOISE_THRESHOLD = 0.42f;

// Steep slopes and high peaks: exposed stone and snow
inline constexpr f32 CLIFF_SLOPE_THRESHOLD = 2.4f;
inline constexpr f32 HIGH_PEAK_SLOPE_THRESHOLD = 1.65f;
inline constexpr i32 HIGH_PEAK_STONE_HEIGHT = 72;
inline constexpr i32 SNOW_BASE_HEIGHT = 92;
inline constexpr f32 SNOW_NOISE_AMPLITUDE = 12.0f;

// ------------------------------------------
// --- Tree spawning and structure rules: ---
// ------------------------------------------

inline constexpr i32 TREE_FOREST_SEED_OFFSET = 503;
inline constexpr f32 TREE_FOREST_SCALE = 260.0f;
inline constexpr i32 TREE_FOREST_OCTAVES = 2;
inline constexpr f32 TREE_FOREST_THRESHOLD = 0.46f;

// Jittered grid cell size (in world blocks) to keep trees from overlapping
inline constexpr i32 TREE_CELL_SIZE = 8;
inline constexpr u32 TREE_SPAWN_PROBABILITY_PERCENT = 55;
inline constexpr f32 TREE_MAX_SLOPE = 1.5f;

inline constexpr i32 TREE_MIN_TRUNK_HEIGHT = 4;
inline constexpr i32 TREE_MAX_TRUNK_HEIGHT = 7;
inline constexpr i32 TREE_CANOPY_RADIUS = 2;
inline constexpr i32 TREE_CANOPY_BELOW_TOP = 2;
inline constexpr i32 TREE_CANOPY_ABOVE_TOP = 2;
inline constexpr i32 TREE_MAX_HEIGHT = TREE_MAX_TRUNK_HEIGHT + TREE_CANOPY_ABOVE_TOP;

} // namespace vl::world_config
