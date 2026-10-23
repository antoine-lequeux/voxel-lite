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

const ChunkGenerator::ColumnInfo& ChunkGenerator::compute_column(i32 cx, i32 cz, u8 lod)
{
    ColumnInfo& info = m_column_cache[column_cache_index(cx, cz)];

    if (m_heights_valid && m_heights_cx == cx && m_heights_cz == cz && m_heights_lod == lod && info.valid && info.cx == cx &&
        info.cz == cz && info.lod == lod)
    {
        return info;
    }

    const u32 shift = static_cast<u32>(lod - 1u);
    const i32 dim = CHUNK_SIZE >> shift;
    const i32 scale = 1 << shift;
    const usize sample_count = static_cast<usize>(dim * dim);

    // Sample at the center of each column
    const f32 center_offset = 0.5f * static_cast<f32>(scale - 1);
    const f32 start_x = (static_cast<f32>(cx * CHUNK_SIZE) + center_offset) * NOISE_FREQUENCY;
    const f32 start_z = (static_cast<f32>(cz * CHUNK_SIZE) + center_offset) * NOISE_FREQUENCY;
    const f32 step = static_cast<f32>(scale) * NOISE_FREQUENCY;

    m_node_graph->GenUniformGrid2D(m_noise_buffer.data(), start_x, start_z, dim, dim, step, step, m_seed);

    i32 min_h = std::numeric_limits<i32>::max();
    i32 max_h = std::numeric_limits<i32>::min();
    for (usize i = 0; i < sample_count; ++i)
    {
        const i32 h = static_cast<i32>(std::floor(TERRAIN_BASE_HEIGHT + (m_noise_buffer[i] * TERRAIN_AMPLITUDE)));
        m_heights[i] = h;
        min_h = std::min(min_h, h);
        max_h = std::max(max_h, h);
    }

    m_heights_cx = cx;
    m_heights_cz = cz;
    m_heights_lod = lod;
    m_heights_valid = true;

    info = ColumnInfo {
        .cx = cx,
        .cz = cz,
        .min_height = min_h,
        .max_height = max_h,
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
    return coord.y * CHUNK_SIZE > info.max_height;
}

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
bool ChunkGenerator::generate(Chunk<Lod>& chunk, bool& out_guaranteed_all_lods, i32& out_column_max_height)
{
    constexpr u32 dim = Chunk<Lod>::DIM;
    constexpr i32 scale = static_cast<i32>(Chunk<Lod>::SCALE);

    const i32 base_y = chunk.coord.y * CHUNK_SIZE;
    const i32 top_y = base_y + CHUNK_SIZE - 1;

    const ColumnInfo& info = compute_column(chunk.coord.x, chunk.coord.z, static_cast<u8>(Lod));
    out_column_max_height = info.max_height;

    if (base_y > info.max_height)
    {
        out_guaranteed_all_lods = (Lod == 1u) || (base_y > info.max_height + SURFACE_AIR_MARGIN);
        return false; // 100% air at this LOD
    }

    // Whole chunk is below the deepest dirt layer
    if (top_y < info.min_height - DIRT_DEPTH - scale)
    {
        chunk.fill(BlockType::Stone);
        chunk.recalculate_portal_mask();
        return true;
    }

    chunk.clear();
    bool any_block = false;

    for (u32 z = 0; z < dim; ++z)
    {
        for (u32 x = 0; x < dim; ++x)
        {
            const i32 h = m_heights[z * dim + x];
            if (h < base_y) continue;

            const i32 max_ly = std::min((h - base_y) / scale, static_cast<i32>(dim) - 1);
            for (i32 ly = 0; ly <= max_ly; ++ly)
            {
                const i32 wy_base = base_y + (ly * scale);

                BlockID block = BlockType::Stone;
                if (wy_base + scale > h)
                    block = BlockType::Grass;
                else if (h - wy_base <= DIRT_DEPTH + scale)
                    block = BlockType::Dirt;

                chunk.set_block_unchecked(x, static_cast<u32>(ly), z, block);
            }
            any_block = true;
        }
    }

    if (!any_block)
    {
        out_guaranteed_all_lods = (Lod == 1u) || (base_y > info.max_height + SURFACE_AIR_MARGIN);
        return false;
    }

    chunk.recalculate_portal_mask();
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
