#include "engine/chunkgenerator.hpp"
#include "block.hpp"

#include <algorithm>

namespace vl
{

ChunkGenerator::ChunkGenerator(i32 seed) : m_noise_buffer(CHUNK_SIZE * CHUNK_SIZE), m_seed(seed)
{
    auto fn_fractal = FastNoise::New<FastNoise::FractalFBm>();
    auto fn_simplex = FastNoise::New<FastNoise::Simplex>();

    fn_fractal->SetSource(fn_simplex);
    fn_fractal->SetOctaveCount(5);

    m_node_graph = fn_fractal;
}

void ChunkGenerator::generate(Chunk& chunk)
{
    chunk.clear();

    f32 frequency = 0.2f;

    // Scale the chunk world offsets into the noise coordinate space
    f32 start_x = static_cast<f32>(chunk.coord.x * CHUNK_SIZE) * frequency;
    f32 start_z = static_cast<f32>(chunk.coord.z * CHUNK_SIZE) * frequency;

    m_node_graph->GenUniformGrid2D(m_noise_buffer.data(), start_x, start_z, CHUNK_SIZE, CHUNK_SIZE, frequency, frequency, m_seed);

    for (u32 x = 0; x < CHUNK_SIZE; ++x)
    {
        for (u32 z = 0; z < CHUNK_SIZE; ++z)
        {
            f32 raw_noise = m_noise_buffer[z * CHUNK_SIZE + x];
            f32 mapped_height = 16.0f + (raw_noise * 16.0f);

            u32 max_y = std::clamp(static_cast<u32>(std::max(0.0f, mapped_height)), 0u, static_cast<u32>(CHUNK_SIZE - 1));

            for (u32 y = 0; y <= max_y; ++y)
            {
                BlockID block = BlockType::Dirt;
                if (y == max_y)
                    block = BlockType::Grass;
                else if (max_y - y > 4)
                    block = BlockType::Stone;

                chunk.set_block_unchecked(x, y, z, block);
            }
        }
    }

    chunk.recalculate_portal_mask();
}

} // namespace vl