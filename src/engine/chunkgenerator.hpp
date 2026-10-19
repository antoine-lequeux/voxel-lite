#pragma once

#include "engine/chunk.hpp"

#include <FastNoise/FastNoise.h>
#include <vector>

namespace vl
{

class ChunkGenerator
{
public:

    ChunkGenerator(i32 seed = 0);

    void generate(Chunk& chunk);

private:

    FastNoise::SmartNode<> m_node_graph;
    std::vector<f32> m_noise_buffer;
    i32 m_seed;
};

} // namespace vl