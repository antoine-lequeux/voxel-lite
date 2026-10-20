#include "engine/chunkgenerator.hpp"
#include "block.hpp"

namespace vl
{

namespace
{

constexpr f32 NOISE_FREQUENCY = 0.08f;

constexpr f32 TERRAIN_BASE_HEIGHT = 100.0f;
constexpr f32 TERRAIN_AMPLITUDE = 200.0f;

constexpr i32 DIRT_DEPTH = 4;

} // namespace

ChunkGenerator::ChunkGenerator(i32 seed)
    : m_noise_buffer(CHUNK_AREA), m_heights(CHUNK_AREA), m_column_cache(COLUMN_CACHE_SIZE * COLUMN_CACHE_SIZE), m_seed(seed)
{
    auto fn_fractal = FastNoise::New<FastNoise::FractalFBm>();
    auto fn_simplex = FastNoise::New<FastNoise::Simplex>();

    fn_fractal->SetSource(fn_simplex);
    fn_fractal->SetOctaveCount(8);

    m_node_graph = fn_fractal;
}

const ChunkGenerator::ColumnInfo& ChunkGenerator::compute_column(i32 cx, i32 cz)
{
    ColumnInfo& info = m_column_cache[column_cache_index(cx, cz)];

    if (m_heights_valid && m_heights_cx == cx && m_heights_cz == cz && info.valid && info.cx == cx && info.cz == cz) return info;

    // Scale the chunk world offsets into the noise coordinate space
    const f32 start_x = static_cast<f32>(cx * CHUNK_SIZE) * NOISE_FREQUENCY;
    const f32 start_z = static_cast<f32>(cz * CHUNK_SIZE) * NOISE_FREQUENCY;

    m_node_graph->GenUniformGrid2D(
        m_noise_buffer.data(), start_x, start_z, CHUNK_SIZE, CHUNK_SIZE, NOISE_FREQUENCY, NOISE_FREQUENCY, m_seed
    );

    i32 min_h = std::numeric_limits<i32>::max();
    i32 max_h = std::numeric_limits<i32>::min();
    for (usize i = 0; i < static_cast<usize>(CHUNK_AREA); ++i)
    {
        const i32 h = static_cast<i32>(std::floor(TERRAIN_BASE_HEIGHT + (m_noise_buffer[i] * TERRAIN_AMPLITUDE)));
        m_heights[i] = h;
        min_h = std::min(min_h, h);
        max_h = std::max(max_h, h);
    }

    m_heights_cx = cx;
    m_heights_cz = cz;
    m_heights_valid = true;

    info = ColumnInfo {.cx = cx, .cz = cz, .min_height = min_h, .max_height = max_h, .valid = true};
    return info;
}

const ChunkGenerator::ColumnInfo& ChunkGenerator::get_column(i32 cx, i32 cz)
{
    const ColumnInfo& info = m_column_cache[column_cache_index(cx, cz)];
    if (info.valid && info.cx == cx && info.cz == cz) return info;
    return compute_column(cx, cz);
}

bool ChunkGenerator::is_chunk_empty(const glm::ivec3& coord)
{
    const ColumnInfo& info = get_column(coord.x, coord.z);
    return coord.y * CHUNK_SIZE > info.max_height;
}

bool ChunkGenerator::generate(Chunk& chunk)
{
    const ColumnInfo& info = compute_column(chunk.coord.x, chunk.coord.z);

    const i32 base_y = chunk.coord.y * CHUNK_SIZE;
    const i32 top_y = base_y + CHUNK_SIZE - 1;

    if (base_y > info.max_height) return false; // 100% air

    // Whole chunk is below the deepest dirt layer
    if (top_y < info.min_height - DIRT_DEPTH)
    {
        chunk.fill(BlockType::Stone);
        chunk.recalculate_portal_mask();
        return true;
    }

    chunk.clear();
    bool any_block = false;

    for (u32 z = 0; z < CHUNK_SIZE; ++z)
    {
        for (u32 x = 0; x < CHUNK_SIZE; ++x)
        {
            const i32 h = m_heights[z * CHUNK_SIZE + x];
            if (h < base_y) continue;

            const i32 local_max = std::min(h - base_y, CHUNK_SIZE - 1);
            for (i32 ly = 0; ly <= local_max; ++ly)
            {
                const i32 wy = base_y + ly;

                BlockID block = BlockType::Dirt;
                if (wy == h)
                    block = BlockType::Grass;
                else if (h - wy > DIRT_DEPTH)
                    block = BlockType::Stone;

                chunk.set_block_unchecked(x, static_cast<u32>(ly), z, block);
            }
            any_block = true;
        }
    }

    chunk.recalculate_portal_mask();
    return any_block;
}

} // namespace vl