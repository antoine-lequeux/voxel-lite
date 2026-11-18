#pragma once

#include "engine/chunk.hpp"
#include "noise_nodes.hpp"
#include "terrain_config.hpp"
#include "tree_generator.hpp"

#include <array>
#include <vector>

namespace vl
{

class ChunkGenerator
{
public:

    static constexpr i32 SURFACE_AIR_MARGIN = 64;

    explicit ChunkGenerator(i32 seed = 0);

    template <u32 Lod = 1>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    [[nodiscard]] bool is_chunk_empty(const glm::ivec3& coord);

    // Generates a chunk in a single column/block pass, returns false if the chunk is 100% air
    template <u32 Lod>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    bool generate(Chunk<Lod>& chunk, bool& out_guaranteed_all_lods, i32& out_column_max_height);

    template <u32 Lod>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    bool generate(Chunk<Lod>& chunk)
    {
        bool unused_all_lods = false;
        i32 unused_max_h = 0;
        return generate(chunk, unused_all_lods, unused_max_h);
    }

private:

    struct ColumnInfo
    {
        i32 cx = 0;
        i32 cz = 0;
        i32 min_height = 0;
        i32 max_height = 0;
        i32 max_ceiling_height = 0;
        u8 lod = 0;
        bool valid = false;
    };

    struct PreparedColumn
    {
        i32 height = 0;
        u32 tree_mask = 0;
        BlockID surface_block = BlockType::Grass;
    };

    static constexpr i32 MAX_PADDED_DIM = CHUNK_SIZE + (2 * world_config::COLUMN_PADDING);
    static constexpr usize MAX_PADDED_AREA = static_cast<usize>(MAX_PADDED_DIM * MAX_PADDED_DIM);
    static constexpr usize MAX_CAVE_BUFFER_SIZE = static_cast<usize>(CHUNK_SIZE * (CHUNK_SIZE + 1) * CHUNK_SIZE);
    static constexpr usize MAX_TREES_PER_CHUNK = 32;

    static constexpr i32 COLUMN_CACHE_BITS = 9;
    static constexpr i32 COLUMN_CACHE_SIZE = 1 << COLUMN_CACHE_BITS;
    static constexpr i32 COLUMN_CACHE_MASK = COLUMN_CACHE_SIZE - 1;

    [[nodiscard]] static constexpr usize column_cache_index(i32 cx, i32 cz)
    {
        return static_cast<usize>(((cz & COLUMN_CACHE_MASK) << COLUMN_CACHE_BITS) | (cx & COLUMN_CACHE_MASK));
    }

    const ColumnInfo& compute_column(i32 cx, i32 cz, u8 lod);
    const ColumnInfo& get_column(i32 cx, i32 cz, u8 lod);

    TerrainNoiseNodes m_nodes {};

    // 2D SIMD noise buffers (padded by COLUMN_PADDING on each side for slope and overhanging trees)
    std::vector<f32> m_continentality_buf {};
    std::vector<f32> m_erosion_buf {};
    std::vector<f32> m_peaks_buf {};
    std::vector<f32> m_patch_buf {};
    std::vector<f32> m_forest_buf {};
    std::vector<i32> m_padded_heights {};

    // 3D SIMD cave noise buffers
    std::vector<f32> m_cave_spaghetti_buf {};
    std::vector<f32> m_cave_large_buf {};

    // Prepared 2D column data and active trees for the current (cx, cz, lod)
    std::vector<PreparedColumn> m_prepared_columns {};
    std::array<TreeInstance, MAX_TREES_PER_CHUNK> m_active_trees {};
    u32 m_active_trees_count = 0;

    std::vector<ColumnInfo> m_column_cache {};
    i32 m_prepared_cx = 0;
    i32 m_prepared_cz = 0;
    u8 m_prepared_lod = 0;
    bool m_prepared_valid = false;
    i32 m_seed = 0;
};

} // namespace vl
