#pragma once

#include "terrain_config.hpp"

#include <algorithm>
#include <cmath>

namespace vl
{

[[nodiscard]] constexpr f32 abs_f32(f32 v)
{
    return (v < 0.0f) ? -v : v;
}

struct SplineSegment
{
    f32 x0 = 0.0f;
    f32 inv_dx = 1.0f;
    f32 y0 = 0.0f;
    f32 y1 = 0.0f;
    f32 m0_dx = 0.0f;
    f32 m1_dx = 0.0f;
};

// Precomputes monotonic cubic Hermite (Fritsch-Carlson) spline segments at compile time
// from the control points in world_config::CONTINENTALITY_SPLINE
[[nodiscard]] consteval auto build_continentality_spline_segments()
{
    constexpr usize N = world_config::CONTINENTALITY_SPLINE.size();
    static_assert(N >= 2, "Continentality spline requires at least 2 control points");

    constexpr usize SEG_COUNT = N - 1;
    std::array<f32, SEG_COUNT> secants {};
    for (usize i = 0; i < SEG_COUNT; ++i)
    {
        const auto& p0 = world_config::CONTINENTALITY_SPLINE[i];
        const auto& p1 = world_config::CONTINENTALITY_SPLINE[i + 1];
        const f32 dx = p1.continentality - p0.continentality;
        secants[i] = (p1.height - p0.height) / dx;
    }

    std::array<f32, N> tangents {};
    tangents[0] = secants[0];
    tangents[N - 1] = secants[SEG_COUNT - 1];

    for (usize i = 1; i < SEG_COUNT; ++i)
    {
        const f32 d0 = secants[i - 1];
        const f32 d1 = secants[i];
        if (d0 * d1 <= 0.0f)
        {
            tangents[i] = 0.0f;
        }
        else
        {
            const f32 avg = 0.5f * (d0 + d1);
            const f32 max_slope = 3.0f * std::min(abs_f32(d0), abs_f32(d1));
            const f32 mag = std::min(abs_f32(avg), max_slope);
            tangents[i] = (avg < 0.0f) ? -mag : mag;
        }
    }

    std::array<SplineSegment, SEG_COUNT> segments {};
    for (usize i = 0; i < SEG_COUNT; ++i)
    {
        const auto& p0 = world_config::CONTINENTALITY_SPLINE[i];
        const auto& p1 = world_config::CONTINENTALITY_SPLINE[i + 1];
        const f32 dx = p1.continentality - p0.continentality;
        segments[i] = SplineSegment {
            .x0 = p0.continentality,
            .inv_dx = 1.0f / dx,
            .y0 = p0.height,
            .y1 = p1.height,
            .m0_dx = tangents[i] * dx,
            .m1_dx = tangents[i + 1] * dx,
        };
    }

    return segments;
}

inline constexpr auto CONTINENTALITY_SEGMENTS = build_continentality_spline_segments();

// Maps continentality in [0, 1] to base terrain elevation in world coordinates
[[nodiscard]] constexpr f32 evaluate_continentality_height(f32 continentality)
{
    const f32 c = std::clamp(continentality, 0.0f, 1.0f);

    usize seg_idx = CONTINENTALITY_SEGMENTS.size() - 1u;
    for (usize i = 0; i < CONTINENTALITY_SEGMENTS.size() - 1u; ++i)
    {
        if (c < CONTINENTALITY_SEGMENTS[i + 1u].x0)
        {
            seg_idx = i;
            break;
        }
    }

    const SplineSegment& seg = CONTINENTALITY_SEGMENTS[seg_idx];
    const f32 t = std::clamp((c - seg.x0) * seg.inv_dx, 0.0f, 1.0f);
    const f32 t2 = t * t;
    const f32 t3 = t2 * t;

    const f32 h00 = (2.0f * t3) - (3.0f * t2) + 1.0f;
    const f32 h10 = t3 - (2.0f * t2) + t;
    const f32 h01 = (-2.0f * t3) + (3.0f * t2);
    const f32 h11 = t3 - t2;

    return (h00 * seg.y0) + (h10 * seg.m0_dx) + (h01 * seg.y1) + (h11 * seg.m1_dx);
}

// Combines continentality, erosion, and peaks (each in [0, 1]) into the final column height
[[nodiscard]] inline i32 compute_column_height(f32 continentality, f32 erosion, f32 peaks)
{
    const f32 c = std::clamp(continentality, 0.0f, 1.0f);
    const f32 e = std::clamp(erosion, 0.0f, 1.0f);
    const f32 p = std::clamp(peaks, 0.0f, 1.0f);

    const f32 base_height = evaluate_continentality_height(c);
    const f32 erosion_reduction = std::pow(e, world_config::EROSION_CURVE_POWER) * world_config::EROSION_MAX_HEIGHT_REDUCTION;
    const f32 eroded_height = base_height - erosion_reduction;
    const f32 peak_bonus = std::pow(p, world_config::PEAKS_CURVE_POWER) * world_config::PEAKS_MAX_HEIGHT_BONUS;

    return static_cast<i32>(std::floor(eroded_height + peak_bonus));
}

// Computes scale-invariant terrain slope (vertical rise per horizontal block)
[[nodiscard]] constexpr f32 compute_column_slope(i32 h_left, i32 h_right, i32 h_back, i32 h_front, i32 step_blocks)
{
    const f32 dx = abs_f32(static_cast<f32>(h_right - h_left));
    const f32 dz = abs_f32(static_cast<f32>(h_front - h_back));
    const f32 inv_span = 1.0f / static_cast<f32>(2 * step_blocks);
    return std::max(dx, dz) * inv_span;
}

} // namespace vl
