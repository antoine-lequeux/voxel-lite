#include "engine/chunk.hpp"

namespace vl
{

namespace
{

struct FaceCornerOffset
{
    u32 dx;
    u32 dy;
    u32 dz;
    u32 uv_id; // bit 0 = U, bit 1 = V
};

inline constexpr std::array<std::array<FaceCornerOffset, 4>, 6> FACE_CORNERS {{
    // Left (-X)
    {{
        {0u, 0u, 0u, 0u},
        {0u, 0u, 1u, 1u},
        {0u, 1u, 1u, 3u},
        {0u, 1u, 0u, 2u},
    }},
    // Right (+X)
    {{
        {1u, 0u, 1u, 0u},
        {1u, 0u, 0u, 1u},
        {1u, 1u, 0u, 3u},
        {1u, 1u, 1u, 2u},
    }},
    // Down (-Y)
    {{
        {0u, 0u, 0u, 0u},
        {1u, 0u, 0u, 1u},
        {1u, 0u, 1u, 3u},
        {0u, 0u, 1u, 2u},
    }},
    // Up (+Y)
    {{
        {0u, 1u, 1u, 0u},
        {1u, 1u, 1u, 1u},
        {1u, 1u, 0u, 3u},
        {0u, 1u, 0u, 2u},
    }},
    // Back (-Z)
    {{
        {1u, 0u, 0u, 0u},
        {0u, 0u, 0u, 1u},
        {0u, 1u, 0u, 3u},
        {1u, 1u, 0u, 2u},
    }},
    // Front (+Z)
    {{
        {0u, 0u, 1u, 0u},
        {1u, 0u, 1u, 1u},
        {1u, 1u, 1u, 3u},
        {0u, 1u, 1u, 2u},
    }},
}};

// Computes voxel corner Ambient Occlusion level [0..3]
[[nodiscard]] u32 compute_corner_ao(
    const Chunk& chunk, const ChunkNeighbors& neighbors, i32 bx, i32 by, i32 bz, Direction dir, const FaceCornerOffset& corner
)
{
    const i32 sx = (corner.dx == 0u) ? -1 : 1;
    const i32 sy = (corner.dy == 0u) ? -1 : 1;
    const i32 sz = (corner.dz == 0u) ? -1 : 1;

    BlockID side1 = BlockType::Air;
    BlockID side2 = BlockType::Air;
    BlockID diag = BlockType::Air;

    switch (dir)
    {
        case Direction::Left:
        case Direction::Right:
        {
            const i32 nx = bx + sx;
            side1 = chunk.sample_block(nx, by + sy, bz, neighbors);
            side2 = chunk.sample_block(nx, by, bz + sz, neighbors);
            diag = chunk.sample_block(nx, by + sy, bz + sz, neighbors);
            break;
        }
        case Direction::Down:
        case Direction::Up:
        {
            const i32 ny = by + sy;
            side1 = chunk.sample_block(bx + sx, ny, bz, neighbors);
            side2 = chunk.sample_block(bx, ny, bz + sz, neighbors);
            diag = chunk.sample_block(bx + sx, ny, bz + sz, neighbors);
            break;
        }
        case Direction::Back:
        case Direction::Front:
        {
            const i32 nz = bz + sz;
            side1 = chunk.sample_block(bx + sx, by, nz, neighbors);
            side2 = chunk.sample_block(bx, by + sy, nz, neighbors);
            diag = chunk.sample_block(bx + sx, by + sy, nz, neighbors);
            break;
        }
    }

    const u32 s1 = is_block_solid(side1) ? 1u : 0u;
    const u32 s2 = is_block_solid(side2) ? 1u : 0u;
    const u32 c = is_block_solid(diag) ? 1u : 0u;

    if (s1 != 0u && s2 != 0u) return 0u;
    return 3u - (s1 + s2 + c);
}

void emit_face(
    ChunkMesh& out_mesh, const Chunk& chunk, const ChunkNeighbors& neighbors, u32 x, u32 y, u32 z, Direction dir, BlockID block_id
)
{
    const auto& corners = FACE_CORNERS[static_cast<usize>(dir)];
    const i32 bx = static_cast<i32>(x);
    const i32 by = static_cast<i32>(y);
    const i32 bz = static_cast<i32>(z);

    std::array<u32, 4> ao {};
    std::array<PackedVertex, 4> quad_verts {};

    for (usize i = 0; i < 4; ++i)
    {
        const FaceCornerOffset& c = corners[i];
        ao[i] = compute_corner_ao(chunk, neighbors, bx, by, bz, dir, c);
        quad_verts[i] = pack_vertex(x + c.dx, y + c.dy, z + c.dz, dir, ao[i], c.uv_id, block_id);
    }

    // Flip the quad's triangle split diagonal when AO is anisotropic to prevent interpolation artifacts
    if (ao[0] + ao[2] < ao[1] + ao[3])
    {
        out_mesh.vertices.push_back(quad_verts[1]);
        out_mesh.vertices.push_back(quad_verts[2]);
        out_mesh.vertices.push_back(quad_verts[3]);
        out_mesh.vertices.push_back(quad_verts[0]);
    }
    else
    {
        out_mesh.vertices.push_back(quad_verts[0]);
        out_mesh.vertices.push_back(quad_verts[1]);
        out_mesh.vertices.push_back(quad_verts[2]);
        out_mesh.vertices.push_back(quad_verts[3]);
    }

    ++out_mesh.quad_count;
}

} // namespace

bool ChunkMesh::upload_to_gpu(vulk::VulkanContext& ctx)
{
    if (vertex_buffer.is_valid())
    {
        ctx.destroy_buffer(vertex_buffer);
        vertex_buffer = {};
    }

    if (vertices.empty() || quad_count == 0u) return true;

    vertex_buffer = ctx.create_buffer(
        vertices.size() * sizeof(PackedVertex), vulk::BufferUsage::Storage | vulk::BufferUsage::ShaderDeviceAddress,
        vulk::MemoryUsage::GpuOnly, vertices.data()
    );

    const vulk::BufferHandle quad_ibo = ctx.get_or_create_quad_index_buffer(quad_count);
    return vertex_buffer.is_valid() && quad_ibo.is_valid();
}

void ChunkMesh::destroy_gpu_resources(vulk::VulkanContext& ctx)
{
    if (vertex_buffer.is_valid())
    {
        ctx.destroy_buffer(vertex_buffer);
        vertex_buffer = {};
    }
}

BlockID Chunk::sample_block(i32 x, i32 y, i32 z, const ChunkNeighbors& neighbors) const
{
    // Inside current chunk
    if (static_cast<u32>(x) < CHUNK_SIZE && static_cast<u32>(y) < CHUNK_SIZE && static_cast<u32>(z) < CHUNK_SIZE)
        return get_block_unchecked(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z));

    // Cross-chunk
    if (static_cast<u32>(y) < CHUNK_SIZE && static_cast<u32>(z) < CHUNK_SIZE)
    {
        if (x < 0 && neighbors.left != nullptr)
            return neighbors.left->get_block(static_cast<u32>(x + CHUNK_SIZE), static_cast<u32>(y), static_cast<u32>(z));
        if (x >= CHUNK_SIZE && neighbors.right != nullptr)
            return neighbors.right->get_block(static_cast<u32>(x - CHUNK_SIZE), static_cast<u32>(y), static_cast<u32>(z));
    }
    if (static_cast<u32>(x) < CHUNK_SIZE && static_cast<u32>(z) < CHUNK_SIZE)
    {
        if (y < 0 && neighbors.down != nullptr)
            return neighbors.down->get_block(static_cast<u32>(x), static_cast<u32>(y + CHUNK_SIZE), static_cast<u32>(z));
        if (y >= CHUNK_SIZE && neighbors.up != nullptr)
            return neighbors.up->get_block(static_cast<u32>(x), static_cast<u32>(y - CHUNK_SIZE), static_cast<u32>(z));
    }
    if (static_cast<u32>(x) < CHUNK_SIZE && static_cast<u32>(y) < CHUNK_SIZE)
    {
        if (z < 0 && neighbors.back != nullptr)
            return neighbors.back->get_block(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z + CHUNK_SIZE));
        if (z >= CHUNK_SIZE && neighbors.front != nullptr)
            return neighbors.front->get_block(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z - CHUNK_SIZE));
    }

    return BlockType::Air;
}

void Chunk::generate_mesh(ChunkMesh& out_mesh, const ChunkNeighbors& neighbors) const
{
    out_mesh.clear();
    out_mesh.coord = coord;

    if (out_mesh.vertices.capacity() < 16384u) out_mesh.vertices.reserve(16384u);

    constexpr u32 last = CHUNK_SIZE - 1u;

    for (u32 z = 0; z < CHUNK_SIZE; ++z)
    {
        for (u32 y = 0; y < CHUNK_SIZE; ++y)
        {
            for (u32 x = 0; x < CHUNK_SIZE; ++x)
            {
                const BlockID current = get_block_unchecked(x, y, z);
                if (current == BlockType::Air) continue;

                // Left (-X)
                const BlockID left_block =
                    (x > 0u) ? get_block_unchecked(x - 1u, y, z)
                             : ((neighbors.left != nullptr) ? neighbors.left->get_block_unchecked(last, y, z) : BlockType::Air);
                if (should_render_face(current, left_block))
                    emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Left, current);

                // Right (+X)
                const BlockID right_block =
                    (x < last) ? get_block_unchecked(x + 1u, y, z)
                               : ((neighbors.right != nullptr) ? neighbors.right->get_block_unchecked(0u, y, z) : BlockType::Air);
                if (should_render_face(current, right_block))
                    emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Right, current);

                // Down (-Y)
                const BlockID down_block =
                    (y > 0u) ? get_block_unchecked(x, y - 1u, z)
                             : ((neighbors.down != nullptr) ? neighbors.down->get_block_unchecked(x, last, z) : BlockType::Air);
                if (should_render_face(current, down_block))
                    emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Down, current);

                // Up (+Y)
                const BlockID up_block =
                    (y < last) ? get_block_unchecked(x, y + 1u, z)
                               : ((neighbors.up != nullptr) ? neighbors.up->get_block_unchecked(x, 0u, z) : BlockType::Air);
                if (should_render_face(current, up_block)) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Up, current);

                // Back (-Z)
                const BlockID back_block =
                    (z > 0u) ? get_block_unchecked(x, y, z - 1u)
                             : ((neighbors.back != nullptr) ? neighbors.back->get_block_unchecked(x, y, last) : BlockType::Air);
                if (should_render_face(current, back_block))
                    emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Back, current);

                // Front (+Z)
                const BlockID front_block =
                    (z < last) ? get_block_unchecked(x, y, z + 1u)
                               : ((neighbors.front != nullptr) ? neighbors.front->get_block_unchecked(x, y, 0u) : BlockType::Air);
                if (should_render_face(current, front_block))
                    emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Front, current);
            }
        }
    }
}

} // namespace vl