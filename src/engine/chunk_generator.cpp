#include "engine/chunk_generator.hpp"
#include "block.hpp"
#include "engine/chunk.hpp"

namespace vl
{

void ChunkGenerator::generate(Chunk& chunk)
{
    chunk.clear();

    for (u32 x = 0; x < CHUNK_SIZE; ++x)
    {
        for (u32 z = 0; z < CHUNK_SIZE; ++z)
        {
            u32 max_y = (x + z) / 2;
            for (u32 y = 0; y <= max_y; ++y)
            {
                BlockID block = BlockType::Dirt;
                if (y == max_y)
                    block = BlockType::Grass;
                else if (max_y - y > 4)
                    block = BlockType::Stone;
                chunk.set_block(x, y, z, block);
            }
        }
    }
}

} // namespace vl
