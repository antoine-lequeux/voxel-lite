#pragma once

#include "chunk.hpp"

#include <FastNoise/FastNoise.h>

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

    // Returns false if the chunk is 100% air
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
        u8 lod = 0;
        bool valid = false;
    };

    static constexpr i32 COLUMN_CACHE_BITS = 9;
    static constexpr i32 COLUMN_CACHE_SIZE = 1 << COLUMN_CACHE_BITS;
    static constexpr i32 COLUMN_CACHE_MASK = COLUMN_CACHE_SIZE - 1;

    [[nodiscard]] static constexpr usize column_cache_index(i32 cx, i32 cz)
    {
        return static_cast<usize>(((cz & COLUMN_CACHE_MASK) << COLUMN_CACHE_BITS) | (cx & COLUMN_CACHE_MASK));
    }

    const ColumnInfo& compute_column(i32 cx, i32 cz, u8 lod);
    const ColumnInfo& get_column(i32 cx, i32 cz, u8 lod);

    FastNoise::SmartNode<> m_node_graph {};
    std::vector<f32> m_noise_buffer {};
    std::vector<i32> m_heights {};
    std::vector<ColumnInfo> m_column_cache {};
    i32 m_heights_cx = 0;
    i32 m_heights_cz = 0;
    u8 m_heights_lod = 0;
    bool m_heights_valid = false;
    i32 m_seed = 0;
};

} // namespace vl
