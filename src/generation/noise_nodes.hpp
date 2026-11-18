#pragma once

#include "terrain_config.hpp"

#include <FastNoise/FastNoise.h>

namespace vl
{

struct TerrainNoiseNodes
{
    FastNoise::SmartNode<> continentality_2d {};
    FastNoise::SmartNode<> erosion_2d {};
    FastNoise::SmartNode<> peaks_2d {};
    FastNoise::SmartNode<> surface_patch_2d {};
    FastNoise::SmartNode<> forest_2d {};
    FastNoise::SmartNode<> cave_spaghetti_3d {};
    FastNoise::SmartNode<> cave_large_3d {};

    [[nodiscard]] static FastNoise::SmartNode<> make_continentality_node()
    {
        auto simplex = FastNoise::New<FastNoise::SuperSimplex>();
        simplex->SetScale(world_config::CONT_FEATURE_SCALE);
        simplex->SetSeedOffset(world_config::CONT_SEED_OFFSET);

        auto fbm = FastNoise::New<FastNoise::FractalFBm>();
        fbm->SetSource(simplex);
        fbm->SetOctaveCount(world_config::CONT_OCTAVES);
        fbm->SetGain(world_config::CONT_GAIN);
        fbm->SetLacunarity(world_config::CONT_LACUNARITY);

        auto warp = FastNoise::New<FastNoise::DomainWarpGradient>();
        warp->SetSource(fbm);
        warp->SetScale(world_config::CONT_WARP_SCALE);
        warp->SetWarpAmplitude(world_config::CONT_WARP_AMPLITUDE);
        warp->SetSeedOffset(world_config::CONT_SEED_OFFSET + 17);

        auto remap = FastNoise::New<FastNoise::Remap>();
        remap->SetSource(warp);
        remap->SetFromMin(world_config::CONT_RAW_MIN);
        remap->SetFromMax(world_config::CONT_RAW_MAX);
        remap->SetToMin(0.0f);
        remap->SetToMax(1.0f);
        remap->SetClampOutput(true);

        return remap;
    }

    [[nodiscard]] static FastNoise::SmartNode<> make_erosion_node()
    {
        auto simplex = FastNoise::New<FastNoise::Simplex>();
        simplex->SetScale(world_config::EROSION_FEATURE_SCALE);
        simplex->SetSeedOffset(world_config::EROSION_SEED_OFFSET);

        auto ridged = FastNoise::New<FastNoise::FractalRidged>();
        ridged->SetSource(simplex);
        ridged->SetOctaveCount(world_config::EROSION_OCTAVES);
        ridged->SetGain(world_config::EROSION_GAIN);
        ridged->SetLacunarity(world_config::EROSION_LACUNARITY);
        ridged->SetWeightedStrength(world_config::EROSION_WEIGHTED_STRENGTH);

        auto warp = FastNoise::New<FastNoise::DomainWarpGradient>();
        warp->SetSource(ridged);
        warp->SetScale(world_config::EROSION_WARP_SCALE);
        warp->SetWarpAmplitude(world_config::EROSION_WARP_AMPLITUDE);
        warp->SetSeedOffset(world_config::EROSION_SEED_OFFSET + 23);

        auto warp_fractal = FastNoise::New<FastNoise::DomainWarpFractalProgressive>();
        warp_fractal->SetSource(warp);
        warp_fractal->SetOctaveCount(2);
        warp_fractal->SetGain(0.5f);
        warp_fractal->SetLacunarity(2.0f);

        auto remap = FastNoise::New<FastNoise::Remap>();
        remap->SetSource(warp_fractal);
        remap->SetFromMin(world_config::EROSION_RAW_MIN);
        remap->SetFromMax(world_config::EROSION_RAW_MAX);
        remap->SetToMin(0.0f);
        remap->SetToMax(1.0f);
        remap->SetClampOutput(true);

        return remap;
    }

    [[nodiscard]] static FastNoise::SmartNode<> make_peaks_node()
    {
        // Isolated circular peaks (EuclideanSquared cellular distance domes)
        auto cell_isolated = FastNoise::New<FastNoise::CellularDistance>();
        cell_isolated->SetScale(world_config::PEAKS_ISOLATED_CELL_SCALE);
        cell_isolated->SetSeedOffset(world_config::PEAKS_ISOLATED_SEED_OFFSET);
        cell_isolated->SetDistanceFunction(FastNoise::DistanceFunction::EuclideanSquared);
        cell_isolated->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        cell_isolated->SetGridJitter(world_config::PEAKS_GRID_JITTER);
        cell_isolated->SetSizeJitter(world_config::PEAKS_SIZE_JITTER);
        cell_isolated->SetOutputMin(0.0f);
        cell_isolated->SetOutputMax(1.0f);

        // Invert [PEAKS_CIRCLE_RADIUS, 0.0] -> [0.0, 1.0] so cell centers are 1.0 and exteriors clamp to 0.0
        auto isolated_cones = FastNoise::New<FastNoise::Remap>();
        isolated_cones->SetSource(cell_isolated);
        isolated_cones->SetFromMin(world_config::PEAKS_CIRCLE_RADIUS);
        isolated_cones->SetFromMax(0.0f);
        isolated_cones->SetToMin(0.0f);
        isolated_cones->SetToMax(1.0f);
        isolated_cones->SetClampOutput(true);

        // Chains / ranges of circular peaks masked by a ridged range belt
        auto cell_range = FastNoise::New<FastNoise::CellularDistance>();
        cell_range->SetScale(world_config::PEAKS_RANGE_CELL_SCALE);
        cell_range->SetSeedOffset(world_config::PEAKS_RANGE_SEED_OFFSET);
        cell_range->SetDistanceFunction(FastNoise::DistanceFunction::EuclideanSquared);
        cell_range->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);
        cell_range->SetGridJitter(world_config::PEAKS_GRID_JITTER);
        cell_range->SetSizeJitter(world_config::PEAKS_SIZE_JITTER);
        cell_range->SetOutputMin(0.0f);
        cell_range->SetOutputMax(1.0f);

        auto range_cones = FastNoise::New<FastNoise::Remap>();
        range_cones->SetSource(cell_range);
        range_cones->SetFromMin(world_config::PEAKS_CIRCLE_RADIUS * 1.25f);
        range_cones->SetFromMax(0.0f);
        range_cones->SetToMin(0.0f);
        range_cones->SetToMax(1.0f);
        range_cones->SetClampOutput(true);

        auto mask_simplex = FastNoise::New<FastNoise::Simplex>();
        mask_simplex->SetScale(world_config::PEAKS_RANGE_MASK_SCALE);
        mask_simplex->SetSeedOffset(world_config::PEAKS_MASK_SEED_OFFSET);

        auto mask_ridged = FastNoise::New<FastNoise::FractalRidged>();
        mask_ridged->SetSource(mask_simplex);
        mask_ridged->SetOctaveCount(world_config::PEAKS_RANGE_MASK_OCTAVES);
        mask_ridged->SetGain(0.5f);
        mask_ridged->SetLacunarity(2.0f);

        auto mask_remap = FastNoise::New<FastNoise::Remap>();
        mask_remap->SetSource(mask_ridged);
        mask_remap->SetFromMin(0.15f);
        mask_remap->SetFromMax(1.05f);
        mask_remap->SetToMin(0.0f);
        mask_remap->SetToMax(1.0f);
        mask_remap->SetClampOutput(true);

        auto masked_ranges = FastNoise::New<FastNoise::Multiply>();
        masked_ranges->SetLHS(range_cones);
        masked_ranges->SetRHS(mask_remap);

        // Combine isolated circular peaks and mountain range circular peaks
        auto combined = FastNoise::New<FastNoise::MaxSmooth>();
        combined->SetLHS(isolated_cones);
        combined->SetRHS(masked_ranges);
        combined->SetSmoothness(world_config::PEAKS_BLEND_SMOOTHNESS);

        // Domain warp so circular regions have organic outlines
        auto warped = FastNoise::New<FastNoise::DomainWarpGradient>();
        warped->SetSource(combined);
        warped->SetScale(world_config::PEAKS_WARP_SCALE);
        warped->SetWarpAmplitude(world_config::PEAKS_WARP_AMPLITUDE);
        warped->SetSeedOffset(world_config::PEAKS_WARP_SEED_OFFSET);

        // Modulate peak sharpness (pointy vs rounded) and plateau capping (sometimes a bit flat)
        auto exponent_noise = FastNoise::New<FastNoise::Simplex>();
        exponent_noise->SetScale(world_config::PEAKS_SHAPE_SELECTOR_SCALE);
        exponent_noise->SetSeedOffset(world_config::PEAKS_SHAPE_SEED_OFFSET);
        exponent_noise->SetOutputMin(0.75f);
        exponent_noise->SetOutputMax(world_config::PEAKS_POINTY_EXPONENT);

        auto pointy_shaped = FastNoise::New<FastNoise::PowFloat>();
        pointy_shaped->SetValue(warped);
        pointy_shaped->SetPow(exponent_noise);

        auto plateau_cap_noise = FastNoise::New<FastNoise::Simplex>();
        plateau_cap_noise->SetScale(world_config::PEAKS_SHAPE_SELECTOR_SCALE);
        plateau_cap_noise->SetSeedOffset(world_config::PEAKS_SHAPE_SEED_OFFSET + 41);
        plateau_cap_noise->SetOutputMin(world_config::PEAKS_FLAT_PLATEAU_CAP);
        plateau_cap_noise->SetOutputMax(1.30f);

        auto flat_or_pointy = FastNoise::New<FastNoise::MinSmooth>();
        flat_or_pointy->SetLHS(pointy_shaped);
        flat_or_pointy->SetRHS(plateau_cap_noise);
        flat_or_pointy->SetSmoothness(world_config::PEAKS_FLAT_SMOOTHNESS);

        auto final_remap = FastNoise::New<FastNoise::Remap>();
        final_remap->SetSource(flat_or_pointy);
        final_remap->SetFromMin(0.0f);
        final_remap->SetFromMax(1.0f);
        final_remap->SetToMin(0.0f);
        final_remap->SetToMax(1.0f);
        final_remap->SetClampOutput(true);

        return final_remap;
    }

    [[nodiscard]] static FastNoise::SmartNode<> make_surface_patch_node()
    {
        auto simplex = FastNoise::New<FastNoise::Simplex>();
        simplex->SetScale(world_config::SURFACE_PATCH_SCALE);
        simplex->SetSeedOffset(world_config::SURFACE_PATCH_SEED_OFFSET);

        auto fbm = FastNoise::New<FastNoise::FractalFBm>();
        fbm->SetSource(simplex);
        fbm->SetOctaveCount(world_config::SURFACE_PATCH_OCTAVES);
        fbm->SetGain(0.5f);
        fbm->SetLacunarity(2.0f);

        auto remap = FastNoise::New<FastNoise::Remap>();
        remap->SetSource(fbm);
        remap->SetFromMin(-1.0f);
        remap->SetFromMax(1.0f);
        remap->SetToMin(0.0f);
        remap->SetToMax(1.0f);
        remap->SetClampOutput(true);

        return remap;
    }

    [[nodiscard]] static FastNoise::SmartNode<> make_forest_node()
    {
        auto simplex = FastNoise::New<FastNoise::Simplex>();
        simplex->SetScale(world_config::TREE_FOREST_SCALE);
        simplex->SetSeedOffset(world_config::TREE_FOREST_SEED_OFFSET);

        auto fbm = FastNoise::New<FastNoise::FractalFBm>();
        fbm->SetSource(simplex);
        fbm->SetOctaveCount(world_config::TREE_FOREST_OCTAVES);
        fbm->SetGain(0.5f);
        fbm->SetLacunarity(2.0f);

        auto remap = FastNoise::New<FastNoise::Remap>();
        remap->SetSource(fbm);
        remap->SetFromMin(-1.0f);
        remap->SetFromMax(1.0f);
        remap->SetToMin(0.0f);
        remap->SetToMax(1.0f);
        remap->SetClampOutput(true);

        return remap;
    }

    // Builds a 3D cave noise node with continuous high-value tubes (plateauing at 0.50) and
    // compact chambers (reaching up to 1.0) surrounded by vast volumes of 0.0
    [[nodiscard]] static FastNoise::SmartNode<> make_cave_3d_node(i32 seed_offset)
    {
        auto make_ridged_sheet = [](i32 offset) -> FastNoise::SmartNode<> {
            auto simplex = FastNoise::New<FastNoise::Simplex>();
            simplex->SetScale(world_config::CAVE_FEATURE_SCALE);
            simplex->SetSeedOffset(offset);

            auto ridged = FastNoise::New<FastNoise::FractalRidged>();
            ridged->SetSource(simplex);
            ridged->SetOctaveCount(world_config::CAVE_OCTAVES);
            ridged->SetGain(world_config::CAVE_GAIN);
            ridged->SetLacunarity(world_config::CAVE_LACUNARITY);
            ridged->SetWeightedStrength(world_config::CAVE_WEIGHTED_STRENGTH);

            return ridged;
        };

        auto sheet_a = make_ridged_sheet(seed_offset);
        auto sheet_b = make_ridged_sheet(seed_offset + world_config::CAVE_SECOND_SHEET_SEED_OFFSET);

        // Intersection of two 3D ridged sheets forms continuous 1D curves (cylindrical tubes in 3D)
        auto tube_core = FastNoise::New<FastNoise::Min>();
        tube_core->SetLHS(sheet_a);
        tube_core->SetRHS(sheet_b);

        // Slow 3D regional bias so caves cluster into localized networks with wide solid stone expanses
        auto region_bias = FastNoise::New<FastNoise::Simplex>();
        region_bias->SetScale(world_config::CAVE_REGION_SCALE);
        region_bias->SetSeedOffset(seed_offset + world_config::CAVE_REGION_SEED_OFFSET);
        region_bias->SetOutputMin(world_config::CAVE_REGION_BIAS_MIN);
        region_bias->SetOutputMax(world_config::CAVE_REGION_BIAS_MAX);

        auto modulated_tube = FastNoise::New<FastNoise::Add>();
        modulated_tube->SetLHS(tube_core);
        modulated_tube->SetRHS(region_bias);

        // Map tube ridge [CAVE_TUBE_RAW_MIN, CAVE_TUBE_RAW_MAX] -> [0.0, 0.50] (clamped)
        // so the entire interior of the tube stays at around 0.50 without leaving a solid core pocket
        auto remap_tube = FastNoise::New<FastNoise::Remap>();
        remap_tube->SetSource(modulated_tube);
        remap_tube->SetFromMin(world_config::CAVE_TUBE_RAW_MIN);
        remap_tube->SetFromMax(world_config::CAVE_TUBE_RAW_MAX);
        remap_tube->SetToMin(0.0f);
        remap_tube->SetToMax(world_config::SPAGHETTI_CAVE_CENTER);
        remap_tube->SetClampOutput(true);

        // Map extreme hotspots [CAVE_CHAMBER_RAW_MIN, CAVE_CHAMBER_RAW_MAX] -> [0.0, 0.50] (clamped)
        // so rare chamber centers rise above 0.80 up to 1.00
        auto remap_chamber = FastNoise::New<FastNoise::Remap>();
        remap_chamber->SetSource(modulated_tube);
        remap_chamber->SetFromMin(world_config::CAVE_CHAMBER_RAW_MIN);
        remap_chamber->SetFromMax(world_config::CAVE_CHAMBER_RAW_MAX);
        remap_chamber->SetToMin(0.0f);
        remap_chamber->SetToMax(1.0f - world_config::SPAGHETTI_CAVE_CENTER);
        remap_chamber->SetClampOutput(true);

        auto final_cave = FastNoise::New<FastNoise::Add>();
        final_cave->SetLHS(remap_tube);
        final_cave->SetRHS(remap_chamber);

        return final_cave;
    }

    [[nodiscard]] static TerrainNoiseNodes create_all()
    {
        return TerrainNoiseNodes {
            .continentality_2d = make_continentality_node(),
            .erosion_2d = make_erosion_node(),
            .peaks_2d = make_peaks_node(),
            .surface_patch_2d = make_surface_patch_node(),
            .forest_2d = make_forest_node(),
            .cave_spaghetti_3d = make_cave_3d_node(world_config::CAVE_SPAGHETTI_SEED_OFFSET),
            .cave_large_3d = make_cave_3d_node(world_config::CAVE_LARGE_SEED_OFFSET),
        };
    }
};

} // namespace vl
