#include "chunk.hpp"

#include <algorithm>

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

inline constexpr std::array<std::array<FaceCornerOffset, 4>, DIRECTION_COUNT> FACE_CORNERS {{
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
template <u32 Lod>
[[nodiscard]] u32 compute_corner_ao(
    const Chunk<Lod>& chunk, const ChunkNeighbors& neighbors, i32 bx, i32 by, i32 bz, Direction dir,
    const FaceCornerOffset& corner
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

template <u32 Lod>
void emit_face(
    ChunkMesh& out_mesh, const Chunk<Lod>& chunk, const ChunkNeighbors& neighbors, u32 x, u32 y, u32 z, Direction dir,
    BlockID block_id
)
{
    constexpr u32 scale = Chunk<Lod>::SCALE;
    const auto& corners = FACE_CORNERS[to_index(dir)];
    const i32 bx = static_cast<i32>(x);
    const i32 by = static_cast<i32>(y);
    const i32 bz = static_cast<i32>(z);

    std::array<u32, 4> ao {};
    std::array<PackedVertex, 4> quad_verts {};

    for (usize i = 0; i < 4; ++i)
    {
        const FaceCornerOffset& c = corners[i];
        ao[i] = compute_corner_ao(chunk, neighbors, bx, by, bz, dir, c);
        quad_verts[i] = pack_vertex((x + c.dx) * scale, (y + c.dy) * scale, (z + c.dz) * scale, dir, ao[i], c.uv_id, block_id);
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

ChunkDrawTable& ChunkDrawTable::instance()
{
    static ChunkDrawTable table;
    return table;
}

u32 ChunkDrawTable::update_slot(u32 current_slot, const glm::ivec3& coord, u32 quad_count, u64 device_address)
{
    u32 slot = current_slot;
    if (slot == INVALID_DRAW_SLOT)
    {
        if (!m_free_slots.empty())
        {
            slot = m_free_slots.back();
            m_free_slots.pop_back();
        }
        else
        {
            slot = static_cast<u32>(m_entries.size());
            m_entries.emplace_back();
            m_dirty_flags.push_back(0u);
        }
        ++m_active_count;
    }

    const GpuChunkDraw entry {
        .origin_and_quads =
            glm::ivec4 {
                coord.x * CHUNK_SIZE,
                coord.y * CHUNK_SIZE,
                coord.z * CHUNK_SIZE,
                static_cast<i32>(quad_count),
            },
        .vertex_buffer_address = device_address,
        .reserved = 0u,
    };

    GpuChunkDraw& dst = m_entries[slot];
    if (dst.origin_and_quads != entry.origin_and_quads || dst.vertex_buffer_address != entry.vertex_buffer_address)
    {
        dst = entry;
        mark_dirty(slot);
    }

    return slot;
}

void ChunkDrawTable::free_slot(u32 slot)
{
    if (slot == INVALID_DRAW_SLOT || slot >= m_entries.size()) return;
    if (m_entries[slot].origin_and_quads.w == 0 && m_entries[slot].vertex_buffer_address == 0) return;

    m_entries[slot] = GpuChunkDraw {};
    mark_dirty(slot);
    m_free_slots.push_back(slot);
    if (m_active_count > 0u) --m_active_count;
}

void ChunkDrawTable::reset()
{
    m_entries.clear();
    m_free_slots.clear();
    m_dirty_slots.clear();
    m_dirty_flags.clear();
    m_active_count = 0;
}

void ChunkDrawTable::flush_dirty(vulk::VulkanContext& ctx, vulk::BufferHandle gpu_buffer)
{
    if (m_dirty_slots.empty() || m_entries.empty()) return;

    std::sort(m_dirty_slots.begin(), m_dirty_slots.end());

    const u32 uploaded =
        ctx.upload_buffer_elements(gpu_buffer, m_entries.data(), static_cast<u32>(sizeof(GpuChunkDraw)), m_dirty_slots);

    for (u32 i = 0; i < uploaded; ++i) m_dirty_flags[m_dirty_slots[i]] = 0u;

    if (uploaded >= m_dirty_slots.size())
        m_dirty_slots.clear();
    else
        m_dirty_slots.erase(m_dirty_slots.begin(), m_dirty_slots.begin() + uploaded);
}

void ChunkDrawTable::mark_dirty(u32 slot)
{
    if (m_dirty_flags[slot] == 0u)
    {
        m_dirty_flags[slot] = 1u;
        m_dirty_slots.push_back(slot);
    }
}

bool ChunkMesh::upload_to_gpu(vulk::VulkanContext& ctx)
{
    if (vertices.empty() || quad_count == 0u)
    {
        destroy_gpu_resources(ctx);
        return true;
    }

    const u64 required_bytes = vertices.size() * sizeof(PackedVertex);

    // Reuse the existing GPU buffer when remeshing if it is already large enough
    if (vertex_buffer.is_valid() && buffer_capacity_bytes >= required_bytes)
    {
        ctx.upload_buffer_data(vertex_buffer, 0, vertices.data(), static_cast<usize>(required_bytes));
        draw_slot = ChunkDrawTable::instance().update_slot(draw_slot, coord, quad_count, device_address);
        const vulk::BufferHandle quad_ibo = ctx.get_or_create_quad_index_buffer(quad_count);
        return quad_ibo.is_valid();
    }

    if (vertex_buffer.is_valid())
    {
        ctx.destroy_buffer(vertex_buffer);
        vertex_buffer = {};
        device_address = 0;
        buffer_capacity_bytes = 0;
    }

    vertex_buffer = ctx.create_buffer(
        required_bytes, vulk::BufferUsage::Storage | vulk::BufferUsage::ShaderDeviceAddress, vulk::MemoryUsage::GpuOnly,
        vertices.data()
    );

    if (vertex_buffer.is_valid())
    {
        device_address = ctx.get_buffer_device_address(vertex_buffer);
        buffer_capacity_bytes = required_bytes;
        draw_slot = ChunkDrawTable::instance().update_slot(draw_slot, coord, quad_count, device_address);
    }
    else if (draw_slot != INVALID_DRAW_SLOT)
    {
        ChunkDrawTable::instance().free_slot(draw_slot);
        draw_slot = INVALID_DRAW_SLOT;
    }

    const vulk::BufferHandle quad_ibo = ctx.get_or_create_quad_index_buffer(quad_count);
    return vertex_buffer.is_valid() && quad_ibo.is_valid();
}

void ChunkMesh::destroy_gpu_resources(vulk::VulkanContext& ctx)
{
    if (draw_slot != INVALID_DRAW_SLOT)
    {
        ChunkDrawTable::instance().free_slot(draw_slot);
        draw_slot = INVALID_DRAW_SLOT;
    }

    if (vertex_buffer.is_valid())
    {
        ctx.destroy_buffer(vertex_buffer);
        vertex_buffer = {};
        device_address = 0;
        buffer_capacity_bytes = 0;
    }
}

BlockID NeighborChunk::sample_meter(u32 x_m, u32 y_m, u32 z_m) const
{
    if (kind == Kind::Air) return BlockType::Air;
    if (kind == Kind::Solid) return BlockType::Stone;

    switch (lod)
    {
        case 1: return static_cast<const Chunk<1>*>(chunk_ptr)->get_block_at_meter(x_m, y_m, z_m);
        case 2: return static_cast<const Chunk<2>*>(chunk_ptr)->get_block_at_meter(x_m, y_m, z_m);
        case 3: return static_cast<const Chunk<3>*>(chunk_ptr)->get_block_at_meter(x_m, y_m, z_m);
        case 4: return static_cast<const Chunk<4>*>(chunk_ptr)->get_block_at_meter(x_m, y_m, z_m);
        case 5: return static_cast<const Chunk<5>*>(chunk_ptr)->get_block_at_meter(x_m, y_m, z_m);
        default: return BlockType::Air;
    }
}

bool NeighborChunk::should_render_boundary_face(
    BlockID current, u8 self_lod, u32 x_m, u32 y_m, u32 z_m, u32 sx, u32 sy, u32 sz
) const
{
    if (kind == Kind::Air) return should_render_face(current, BlockType::Air);
    if (kind == Kind::Solid) return false;

    // Neighbor macroblocks are the same size or larger than ours, so a single sample covers the entire face
    if (lod >= self_lod) return should_render_face(current, sample_meter(x_m, y_m, z_m));

    // Neighbor has smaller blocks, so we emit the face if it's not entirely covered
    const u32 step = 1u << (lod - 1u);
    const u32 step_x = std::min(step, sx);
    const u32 step_y = std::min(step, sy);
    const u32 step_z = std::min(step, sz);

    for (u32 dz = 0; dz < sz; dz += step_z)
    {
        for (u32 dy = 0; dy < sy; dy += step_y)
        {
            for (u32 dx = 0; dx < sx; dx += step_x)
                if (should_render_face(current, sample_meter(x_m + dx, y_m + dy, z_m + dz))) return true;
        }
    }

    return false;
}

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
BlockID Chunk<Lod>::sample_block(i32 x, i32 y, i32 z, const ChunkNeighbors& neighbors) const
{
    // Inside current chunk
    if (static_cast<u32>(x) < DIM && static_cast<u32>(y) < DIM && static_cast<u32>(z) < DIM)
        return get_block_unchecked(static_cast<u32>(x), static_cast<u32>(y), static_cast<u32>(z));

    constexpr u32 last_meter = static_cast<u32>(CHUNK_SIZE) - 1u;

    // Cross-chunk in meter coordinates
    if (static_cast<u32>(y) < DIM && static_cast<u32>(z) < DIM)
    {
        const u32 y_m = static_cast<u32>(y) * SCALE;
        const u32 z_m = static_cast<u32>(z) * SCALE;
        if (x < 0) return neighbors.left.sample_meter(last_meter, y_m, z_m);
        if (static_cast<u32>(x) >= DIM) return neighbors.right.sample_meter(0u, y_m, z_m);
    }
    if (static_cast<u32>(x) < DIM && static_cast<u32>(z) < DIM)
    {
        const u32 x_m = static_cast<u32>(x) * SCALE;
        const u32 z_m = static_cast<u32>(z) * SCALE;
        if (y < 0) return neighbors.down.sample_meter(x_m, last_meter, z_m);
        if (static_cast<u32>(y) >= DIM) return neighbors.up.sample_meter(x_m, 0u, z_m);
    }
    if (static_cast<u32>(x) < DIM && static_cast<u32>(y) < DIM)
    {
        const u32 x_m = static_cast<u32>(x) * SCALE;
        const u32 y_m = static_cast<u32>(y) * SCALE;
        if (z < 0) return neighbors.back.sample_meter(x_m, y_m, last_meter);
        if (static_cast<u32>(z) >= DIM) return neighbors.front.sample_meter(x_m, y_m, 0u);
    }

    return BlockType::Air;
}

template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
void Chunk<Lod>::generate_mesh(ChunkMesh& out_mesh, const ChunkNeighbors& neighbors) const
{
    out_mesh.clear();
    out_mesh.coord = coord;

    constexpr u32 reserve_verts = std::max(64u, 16384u >> ((Lod - 1u) * 2u));
    if (out_mesh.vertices.capacity() < reserve_verts) out_mesh.vertices.reserve(reserve_verts);

    constexpr u32 last = DIM - 1u;
    constexpr u32 last_meter = static_cast<u32>(CHUNK_SIZE) - 1u;
    constexpr u8 self_lod = static_cast<u8>(Lod);

    for (u32 z = 0; z < DIM; ++z)
    {
        const u32 z_m = z * SCALE;
        for (u32 y = 0; y < DIM; ++y)
        {
            const u32 y_m = y * SCALE;
            for (u32 x = 0; x < DIM; ++x)
            {
                const BlockID current = get_block_unchecked(x, y, z);
                if (current == BlockType::Air) continue;

                const u32 x_m = x * SCALE;

                // Left (-X)
                const bool render_left =
                    (x > 0u)
                        ? should_render_face(current, get_block_unchecked(x - 1u, y, z))
                        : neighbors.left.should_render_boundary_face(current, self_lod, last_meter, y_m, z_m, 1u, SCALE, SCALE);
                if (render_left) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Left, current);

                // Right (+X)
                const bool render_right =
                    (x < last) ? should_render_face(current, get_block_unchecked(x + 1u, y, z))
                               : neighbors.right.should_render_boundary_face(current, self_lod, 0u, y_m, z_m, 1u, SCALE, SCALE);
                if (render_right) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Right, current);

                // Down (-Y)
                const bool render_down =
                    (y > 0u)
                        ? should_render_face(current, get_block_unchecked(x, y - 1u, z))
                        : neighbors.down.should_render_boundary_face(current, self_lod, x_m, last_meter, z_m, SCALE, 1u, SCALE);
                if (render_down) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Down, current);

                // Up (+Y)
                const bool render_up =
                    (y < last) ? should_render_face(current, get_block_unchecked(x, y + 1u, z))
                               : neighbors.up.should_render_boundary_face(current, self_lod, x_m, 0u, z_m, SCALE, 1u, SCALE);
                if (render_up) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Up, current);

                // Back (-Z)
                const bool render_back =
                    (z > 0u)
                        ? should_render_face(current, get_block_unchecked(x, y, z - 1u))
                        : neighbors.back.should_render_boundary_face(current, self_lod, x_m, y_m, last_meter, SCALE, SCALE, 1u);
                if (render_back) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Back, current);

                // Front (+Z)
                const bool render_front =
                    (z < last) ? should_render_face(current, get_block_unchecked(x, y, z + 1u))
                               : neighbors.front.should_render_boundary_face(current, self_lod, x_m, y_m, 0u, SCALE, SCALE, 1u);
                if (render_front) emit_face(out_mesh, *this, neighbors, x, y, z, Direction::Front, current);
            }
        }
    }
}

template class Chunk<1>;
template class Chunk<2>;
template class Chunk<3>;
template class Chunk<4>;
template class Chunk<5>;

} // namespace vl
