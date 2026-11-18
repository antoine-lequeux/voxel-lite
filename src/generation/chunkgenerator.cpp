#include "chunkgenerator.hpp"

#include "engine/block.hpp"
#include "heightmap.hpp"
#include "painter.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace vl
{

ChunkGenerator::ChunkGenerator(i32 seed)
    : m_nodes(TerrainNoiseNodes::create_all()), m_continentality_buf(MAX_PADDED_AREA), m_erosion_buf(MAX_PADDED_AREA),
      m_peaks_buf(MAX_PADDED_AREA), m_patch_buf(MAX_PADDED_AREA), m_forest_buf(MAX_PADDED_AREA),
      m_padded_heights(MAX_PADDED_AREA), m_cave_spaghetti_buf(MAX_CAVE_BUFFER_SIZE), m_cave_large_buf(MAX_CAVE_BUFFER_SIZE),
      m_prepared_columns(CHUNK_AREA), m_column_cache(COLUMN_CACHE_SIZE * COLUMN_CACHE_SIZE), m_seed(seed)
{
}

const ChunkGenerator::ColumnInfo& ChunkGenerator::compute_column(i32 cx, i32 cz, u8 lod)
{
    ColumnInfo& info = m_column_cache[column_cache_index(cx, cz)];

    if (m_prepared_valid && m_prepared_cx == cx && m_prepared_cz == cz && m_prepared_lod == lod && info.valid && info.cx == cx &&
        info.cz == cz && info.lod == lod)
    {
        return info;
    }

    const u32 shift = static_cast<u32>(lod - 1u);
    const i32 dim = CHUNK_SIZE >> shift;
    const i32 scale = 1 << shift;

    constexpr i32 pad = world_config::COLUMN_PADDING;
    const i32 padded_dim = dim + (2 * pad);
    const usize padded_count = static_cast<usize>(padded_dim * padded_dim);

    const i32 chunk_wx = cx * CHUNK_SIZE;
    const i32 chunk_wz = cz * CHUNK_SIZE;
    const f32 start_x = static_cast<f32>(chunk_wx - (pad * scale));
    const f32 start_z = static_cast<f32>(chunk_wz - (pad * scale));
    const f32 step = static_cast<f32>(scale);

    // Pre-generate 2D column noises in SIMD batches
    m_nodes.continentality_2d->GenUniformGrid2D(
        m_continentality_buf.data(), start_x, start_z, padded_dim, padded_dim, step, step, m_seed
    );
    m_nodes.erosion_2d->GenUniformGrid2D(m_erosion_buf.data(), start_x, start_z, padded_dim, padded_dim, step, step, m_seed);
    m_nodes.peaks_2d->GenUniformGrid2D(m_peaks_buf.data(), start_x, start_z, padded_dim, padded_dim, step, step, m_seed);
    m_nodes.surface_patch_2d->GenUniformGrid2D(m_patch_buf.data(), start_x, start_z, padded_dim, padded_dim, step, step, m_seed);

    if (lod <= world_config::MAX_TREE_LOD)
        m_nodes.forest_2d->GenUniformGrid2D(m_forest_buf.data(), start_x, start_z, padded_dim, padded_dim, step, step, m_seed);

    // Evaluate terrain heights across the padded grid from the 3 2D noises
    for (usize i = 0; i < padded_count; ++i)
        m_padded_heights[i] = compute_column_height(m_continentality_buf[i], m_erosion_buf[i], m_peaks_buf[i]);

    i32 min_h = std::numeric_limits<i32>::max();
    i32 max_h = std::numeric_limits<i32>::min();
    const usize stride = static_cast<usize>(padded_dim);

    // Populate the chunk's columns (height + surface block)
    for (i32 z = 0; z < dim; ++z)
    {
        const usize pz = static_cast<usize>(z + pad);
        for (i32 x = 0; x < dim; ++x)
        {
            const usize px = static_cast<usize>(x + pad);
            const usize pidx = (pz * stride) + px;

            const i32 h = m_padded_heights[pidx];
            const f32 slope = compute_column_slope(
                m_padded_heights[pidx - 1u], m_padded_heights[pidx + 1u], m_padded_heights[pidx - stride],
                m_padded_heights[pidx + stride], scale
            );
            const BlockID surface_block = choose_surface_block(h, slope, m_patch_buf[pidx]);

            m_prepared_columns[static_cast<usize>((z * dim) + x)] = PreparedColumn {
                .height = h,
                .tree_mask = 0u,
                .surface_block = surface_block,
            };

            min_h = std::min(min_h, h);
            max_h = std::max(max_h, h);
        }
    }

    // Identify trees in the chunk and its overhanging border margin (only if grass is below)
    m_active_trees_count = 0;
    i32 max_tree_top = std::numeric_limits<i32>::min();

    if (lod <= world_config::MAX_TREE_LOD)
    {
        const i32 canopy_cells = (world_config::TREE_CANOPY_RADIUS + scale - 1) / scale;

        for (i32 pz = 1; pz < padded_dim - 1; ++pz)
        {
            const i32 local_z = pz - pad;
            const i32 wz = chunk_wz + (local_z * scale);

            for (i32 px = 1; px < padded_dim - 1; ++px)
            {
                const i32 local_x = px - pad;
                const i32 wx = chunk_wx + (local_x * scale);
                const usize pidx = (static_cast<usize>(pz) * stride) + static_cast<usize>(px);

                const i32 h = m_padded_heights[pidx];
                const f32 slope = compute_column_slope(
                    m_padded_heights[pidx - 1u], m_padded_heights[pidx + 1u], m_padded_heights[pidx - stride],
                    m_padded_heights[pidx + stride], scale
                );
                const BlockID surface_block = choose_surface_block(h, slope, m_patch_buf[pidx]);

                if (!can_spawn_tree_at_column(wx, wz, slope, m_forest_buf[pidx], surface_block, m_seed)) continue;

                // Ensure the grass surface block directly below the tree was not taken away by a cave
                if (lod <= world_config::MAX_CAVE_LOD)
                {
                    const f32 c2 = m_nodes.cave_large_3d->GenSingle3D(
                        static_cast<f32>(wx), static_cast<f32>(h), static_cast<f32>(wz), m_seed
                    );
                    if (c2 >= world_config::CAVE_ENTRANCE_RARITY_THRESHOLD)
                    {
                        const f32 c1 = m_nodes.cave_spaghetti_3d->GenSingle3D(
                            static_cast<f32>(wx), static_cast<f32>(h), static_cast<f32>(wz), m_seed
                        );
                        if (is_cave_carved(c1, c2, h, 0)) continue;
                    }
                }

                if (m_active_trees_count >= MAX_TREES_PER_CHUNK) break;

                const i32 trunk_h = compute_tree_trunk_height(wx, wz, m_seed);
                const u32 tree_bit = 1u << m_active_trees_count;
                m_active_trees[m_active_trees_count++] = TreeInstance {
                    .root_x = wx,
                    .root_y = h,
                    .root_z = wz,
                    .trunk_height = trunk_h,
                };

                max_tree_top = std::max(max_tree_top, h + trunk_h + world_config::TREE_CANOPY_ABOVE_TOP);

                const i32 min_cx = std::max(0, local_x - canopy_cells);
                const i32 max_cx = std::min(dim - 1, local_x + canopy_cells);
                const i32 min_cz = std::max(0, local_z - canopy_cells);
                const i32 max_cz = std::min(dim - 1, local_z + canopy_cells);

                for (i32 tz = min_cz; tz <= max_cz; ++tz)
                    for (i32 tx = min_cx; tx <= max_cx; ++tx)
                        m_prepared_columns[static_cast<usize>((tz * dim) + tx)].tree_mask |= tree_bit;
            }
        }
    }

    const i32 max_ceiling = std::max({
        max_h + world_config::TREE_MAX_HEIGHT,
        max_tree_top,
        world_config::SEA_LEVEL - 1,
    });

    m_prepared_cx = cx;
    m_prepared_cz = cz;
    m_prepared_lod = lod;
    m_prepared_valid = true;

    info = ColumnInfo {
        .cx = cx,
        .cz = cz,
        .min_height = min_h,
        .max_height = max_h,
        .max_ceiling_height = max_ceiling,
        .lod = lod,
        .valid = true,
    };
    return info;
}

const ChunkGenerator::ColumnInfo& ChunkGenerator::get_column(i32 cx, i32 cz, u8 lod)
{
    const ColumnInfo& info = m_column_cache[column_cache_index(cx, cz)];
    if (info.valid && info.cx == cx && info.cz == cz && info.lod == lod) return info;
    return compute_column(cx, cz, lod);
}

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
bool ChunkGenerator::is_chunk_empty(const glm::ivec3& coord)
{
    const ColumnInfo& info = get_column(coord.x, coord.z, static_cast<u8>(Lod));
    return (coord.y * CHUNK_SIZE) > info.max_ceiling_height;
}

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
bool ChunkGenerator::generate(Chunk<Lod>& chunk, bool& out_guaranteed_all_lods, i32& out_column_max_height)
{
    constexpr u32 dim = Chunk<Lod>::DIM;
    constexpr i32 scale = static_cast<i32>(Chunk<Lod>::SCALE);

    const i32 base_y = chunk.coord.y * CHUNK_SIZE;
    const i32 top_y = base_y + CHUNK_SIZE - 1;

    // Query the 512x512 column summary cache first so early-outs do not recompute 2D column arrays
    const ColumnInfo& info = get_column(chunk.coord.x, chunk.coord.z, static_cast<u8>(Lod));
    out_column_max_height = info.max_ceiling_height;

    // Chunk is completely above terrain, trees, and water
    if (base_y > info.max_ceiling_height)
    {
        out_guaranteed_all_lods = (Lod == 1u) || (base_y > info.max_ceiling_height + SURFACE_AIR_MARGIN);
        return false;
    }

    // Chunk is completely underwater and strictly above the ocean floor
    if (base_y > info.max_height && top_y < world_config::SEA_LEVEL && info.max_height < world_config::SEA_LEVEL)
    {
        chunk.fill(BlockType::Water);
        return true;
    }

    // At distant LODs (where caves are inactive), if chunk is deep solid stone
    if constexpr (Lod > world_config::MAX_CAVE_LOD)
    {
        constexpr i32 MAX_SUBSURFACE_DEPTH = std::max({
            world_config::DIRT_DEPTH,
            world_config::SAND_DEPTH,
            world_config::UNDERWATER_SEDIMENT_DEPTH,
            world_config::SNOW_DEPTH,
        });

        if (top_y < info.min_height - MAX_SUBSURFACE_DEPTH - scale)
        {
            chunk.fill(BlockType::Stone);
            return true;
        }
    }

    // Ensure full per-column surface and tree data is prepared for this (cx, cz, lod)
    compute_column(chunk.coord.x, chunk.coord.z, static_cast<u8>(Lod));

    // Pre-generate the 2 3D cave noise grids in SIMD for the Y slices that overlap ground level
    i32 cave_y_count = 0;
    if constexpr (Lod <= world_config::MAX_CAVE_LOD)
    {
        if (base_y <= info.max_height)
        {
            const i32 max_ground_ly = std::min((info.max_height - base_y) / scale, static_cast<i32>(dim) - 1);
            // Include 1 extra slice (+2) so checking whether (ly + 1) is carved is always in bounds
            cave_y_count = std::min(max_ground_ly + 2, static_cast<i32>(dim) + 1);

            const f32 start_x = static_cast<f32>(chunk.coord.x * CHUNK_SIZE);
            const f32 start_y = static_cast<f32>(base_y);
            const f32 start_z = static_cast<f32>(chunk.coord.z * CHUNK_SIZE);
            const f32 step = static_cast<f32>(scale);

            m_nodes.cave_spaghetti_3d->GenUniformGrid3D(
                m_cave_spaghetti_buf.data(), start_x, start_y, start_z, static_cast<i32>(dim), cave_y_count,
                static_cast<i32>(dim), step, step, step, m_seed
            );

            m_nodes.cave_large_3d->GenUniformGrid3D(
                m_cave_large_buf.data(), start_x, start_y, start_z, static_cast<i32>(dim), cave_y_count, static_cast<i32>(dim),
                step, step, step, m_seed
            );
        }
    }

    chunk.clear();
    bool any_block = false;
    bool any_cave_carved = false;

    const i32 chunk_wx = chunk.coord.x * CHUNK_SIZE;
    const i32 chunk_wz = chunk.coord.z * CHUNK_SIZE;

    // Single pass over each column (x, z) and each vertical block in the column
    for (u32 z = 0; z < dim; ++z)
    {
        const i32 wz = chunk_wz + (static_cast<i32>(z) * scale);

        for (u32 x = 0; x < dim; ++x)
        {
            const i32 wx = chunk_wx + (static_cast<i32>(x) * scale);
            const PreparedColumn& col = m_prepared_columns[(z * dim) + x];
            const i32 col_height = col.height;

            const i32 ground_max_ly =
                (col_height >= base_y) ? std::min((col_height - base_y) / scale, static_cast<i32>(dim) - 1) : -1;

            // Pass A (0 -> col_height): choose between cave (Air) and painted solid block
            if (ground_max_ly >= 0)
            {
                usize cave_idx = ((static_cast<usize>(z) * static_cast<usize>(cave_y_count)) * dim) + x;

                for (i32 ly = 0; ly <= ground_max_ly; ++ly, cave_idx += dim)
                {
                    const i32 wy = base_y + (ly * scale);
                    const i32 depth = col_height - wy;

                    if constexpr (Lod <= world_config::MAX_CAVE_LOD)
                    {
                        if (is_cave_carved(m_cave_spaghetti_buf[cave_idx], m_cave_large_buf[cave_idx], col_height, depth))
                        {
                            any_cave_carved = true;
                            continue;
                        }
                    }

                    bool has_solid_above = (wy + scale <= col_height);
                    if constexpr (Lod <= world_config::MAX_CAVE_LOD)
                    {
                        if (has_solid_above)
                        {
                            const usize above_idx = cave_idx + dim;
                            has_solid_above = !is_cave_carved(
                                m_cave_spaghetti_buf[above_idx], m_cave_large_buf[above_idx], col_height, depth - scale
                            );
                        }
                    }

                    const BlockID block = paint_solid_block(wy, scale, col_height, col.surface_block, has_solid_above);
                    chunk.set_block_unchecked(x, static_cast<u32>(ly), z, block);
                    any_block = true;
                }
            }

            // Pass B (col_height + 1 -> dim - 1): choose between Water / Air and Tree blocks
            const i32 above_start_ly = ground_max_ly + 1;
            if (above_start_ly >= static_cast<i32>(dim)) continue;

            const u32 tree_mask = col.tree_mask;
            const i32 first_above_wy = base_y + (above_start_ly * scale);

            // If this column is entirely above sea level and has no overlapping trees, it remains Air
            if (first_above_wy >= world_config::SEA_LEVEL && tree_mask == 0u) continue;

            for (i32 ly = above_start_ly; ly < static_cast<i32>(dim); ++ly)
            {
                const i32 wy = base_y + (ly * scale);
                BlockID block = (wy < world_config::SEA_LEVEL) ? BlockType::Water : BlockType::Air;

                if constexpr (Lod <= world_config::MAX_TREE_LOD)
                {
                    if (tree_mask != 0u)
                    {
                        for (u32 m = tree_mask; m != 0u; m &= (m - 1u))
                        {
                            const u32 tree_idx = static_cast<u32>(std::countr_zero(m));
                            const BlockID tree_block = sample_tree_voxel(wx, wy, wz, m_active_trees[tree_idx]);
                            if (tree_block == BlockType::Wood)
                            {
                                block = BlockType::Wood;
                                break;
                            }
                            if (tree_block == BlockType::Leaves) block = BlockType::Leaves;
                        }
                    }
                }

                if (block != BlockType::Air)
                {
                    chunk.set_block_unchecked(x, static_cast<u32>(ly), z, block);
                    any_block = true;
                }
            }
        }
    }

    if (!any_block)
    {
        out_guaranteed_all_lods = (Lod == 1u) && (base_y > info.max_ceiling_height);
        return false;
    }

    chunk.recalculate_portal_mask(any_cave_carved);
    return true;
}

template bool ChunkGenerator::is_chunk_empty<1>(const glm::ivec3&);
template bool ChunkGenerator::is_chunk_empty<2>(const glm::ivec3&);
template bool ChunkGenerator::is_chunk_empty<3>(const glm::ivec3&);
template bool ChunkGenerator::is_chunk_empty<4>(const glm::ivec3&);
template bool ChunkGenerator::is_chunk_empty<5>(const glm::ivec3&);

template bool ChunkGenerator::generate<1>(Chunk<1>&, bool&, i32&);
template bool ChunkGenerator::generate<2>(Chunk<2>&, bool&, i32&);
template bool ChunkGenerator::generate<3>(Chunk<3>&, bool&, i32&);
template bool ChunkGenerator::generate<4>(Chunk<4>&, bool&, i32&);
template bool ChunkGenerator::generate<5>(Chunk<5>&, bool&, i32&);

} // namespace vl
