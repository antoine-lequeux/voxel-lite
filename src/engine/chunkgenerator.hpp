#pragma once

#include "engine/chunk.hpp"

#include <FastNoise/FastNoise.h>

namespace vl
{

class ChunkGenerator
{
public:

    ChunkGenerator(i32 seed = 0);

    [[nodiscard]] bool is_chunk_empty(const glm::ivec3& coord);

    // Returns false if the chunk is 100% air
    bool generate(Chunk& chunk);

private:

    struct ColumnInfo
    {
        i32 cx = 0;
        i32 cz = 0;
        i32 min_height = 0;
        i32 max_height = 0;
        bool valid = false;
    };

    static constexpr i32 COLUMN_CACHE_BITS = 8;
    static constexpr i32 COLUMN_CACHE_SIZE = 1 << COLUMN_CACHE_BITS;
    static constexpr i32 COLUMN_CACHE_MASK = COLUMN_CACHE_SIZE - 1;

    [[nodiscard]] static constexpr usize column_cache_index(i32 cx, i32 cz)
    {
        return static_cast<usize>(((cz & COLUMN_CACHE_MASK) << COLUMN_CACHE_BITS) | (cx & COLUMN_CACHE_MASK));
    }

    // Computes the heightmap of a chunk column
    const ColumnInfo& compute_column(i32 cx, i32 cz);
    const ColumnInfo& get_column(i32 cx, i32 cz);

    FastNoise::SmartNode<> m_node_graph;
    std::vector<f32> m_noise_buffer;
    std::vector<i32> m_heights;
    std::vector<ColumnInfo> m_column_cache;
    i32 m_heights_cx = 0;
    i32 m_heights_cz = 0;
    bool m_heights_valid = false;
    i32 m_seed;
};

} // namespace vl