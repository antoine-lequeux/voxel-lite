#pragma once

#include "chunk.hpp"

#include <memory>

namespace vl
{

inline constexpr u32 INVALID_POOL_INDEX = std::numeric_limits<u32>::max();

inline constexpr i32 MIN_CHUNK_Y = -8;
inline constexpr i32 MAX_CHUNK_Y = 31;
inline constexpr i32 WORLD_HEIGHT_CHUNKS = MAX_CHUNK_Y - MIN_CHUNK_Y + 1;

inline constexpr i32 MIN_RENDER_DIST = 4;
inline constexpr i32 MAX_RENDER_DIST = 512;
inline constexpr i32 DEFAULT_RENDER_DIST = 128;

// Half extents (in chunks) of the box around the camera chunk for each LOD
inline constexpr std::array<i32, MAX_LOD> LOD_BOX_RADII = {
    24,              // LOD 1
    48,              // LOD 2
    96,              // LOD 3
    128,             // LOD 4
    MAX_RENDER_DIST, // LOD 5
};

// Computes at compile time the maximum number of chunks that can fit inside an LOD's 3D box
template <u32 Lod>
    requires(Lod >= 1 && Lod <= MAX_LOD)
consteval u32 compute_lod_pool_capacity()
{
    const i32 configured_r = LOD_BOX_RADII[Lod - 1u];
    const i32 r_out = (Lod == MAX_LOD) ? std::max(configured_r, MAX_RENDER_DIST) : configured_r;
    const u32 xz_out = static_cast<u32>((2 * r_out) + 1);
    const u32 y_out = static_cast<u32>(std::min((2 * r_out) + 1, WORLD_HEIGHT_CHUNKS));
    const u32 outer_vol = xz_out * xz_out * y_out;

    if constexpr (Lod == 1)
    {
        return outer_vol + (2u * xz_out * y_out);
    }
    else
    {
        const i32 r_in = LOD_BOX_RADII[Lod - 2u];
        if (r_in >= r_out) return 2u * xz_out * y_out;

        const u32 xz_in = static_cast<u32>((2 * r_in) + 1);
        const u32 min_y_in = static_cast<u32>(std::min(r_in + 1, WORLD_HEIGHT_CHUNKS));
        const u32 min_inner_vol = xz_in * xz_in * min_y_in;
        const u32 transition_slack = 2u * xz_out * y_out;
        return std::max(1u, (outer_vol - min_inner_vol) + transition_slack);
    }
}

template <u32 Lod, u32 MaxChunks = compute_lod_pool_capacity<Lod>()>
    requires(Lod >= 1 && Lod <= MAX_LOD)
class ChunkPool
{
public:

    using ChunkType = Chunk<Lod>;
    static constexpr u32 LOD = Lod;

    ChunkPool()
        : m_chunks(MaxChunks), m_active_slots(std::make_unique<u32[]>(MaxChunks)),
          m_slot_to_active(std::make_unique<u32[]>(MaxChunks)), m_free_slots(std::make_unique<u32[]>(MaxChunks))
    {
        for (u32 i = 0; i < MaxChunks; ++i)
        {
            m_slot_to_active[i] = INVALID_POOL_INDEX;
            m_free_slots[i] = (MaxChunks - 1u) - i;
        }
        m_free_count = MaxChunks;
        m_meshes.reserve(4096);
    }

    [[nodiscard]] constexpr u32 size() const { return m_active_count; }
    [[nodiscard]] constexpr bool empty() const { return m_active_count == 0u; }
    [[nodiscard]] constexpr bool full() const { return m_free_count == 0u; }
    [[nodiscard]] static constexpr u32 capacity() { return MaxChunks; }

    [[nodiscard]] u32 active_slot(u32 active_index) const
    {
        assert(active_index < m_active_count);
        return m_active_slots[active_index];
    }

    [[nodiscard]] u32 allocate(const glm::ivec3& coord)
    {
        assert(m_free_count > 0u && "ChunkPool capacity exceeded");

        const u32 slot = m_free_slots[--m_free_count];

        if (!m_chunks[slot]) m_chunks[slot] = std::make_unique<ChunkType>();
        if (slot >= m_meshes.size()) m_meshes.resize(static_cast<usize>(slot) + 1u);

        ChunkType& chunk = *m_chunks[slot];
        chunk.coord = coord;
        chunk.mesh_dirty = true;

        ChunkMesh& mesh = m_meshes[slot];
        mesh.coord = coord;
        mesh.meshed_once = false;
        mesh.clear();

        const u32 active_idx = m_active_count++;
        m_active_slots[active_idx] = slot;
        m_slot_to_active[slot] = active_idx;

        return slot;
    }

    void free(u32 slot, vulk::VulkanContext& ctx)
    {
        assert(slot < MaxChunks);
        const u32 active_idx = m_slot_to_active[slot];
        if (active_idx == INVALID_POOL_INDEX) return;

        if (slot < m_meshes.size())
        {
            m_meshes[slot].destroy_gpu_resources(ctx);
            m_meshes[slot].clear();
        }

        const u32 last_idx = --m_active_count;
        if (active_idx != last_idx)
        {
            const u32 moved_slot = m_active_slots[last_idx];
            m_active_slots[active_idx] = moved_slot;
            m_slot_to_active[moved_slot] = active_idx;
        }

        m_slot_to_active[slot] = INVALID_POOL_INDEX;
        m_free_slots[m_free_count++] = slot;
    }

    template <typename Self>
    [[nodiscard]] decltype(auto) chunk_at(this Self&& self, u32 slot)
    {
        assert(slot < MaxChunks && self.m_chunks[slot] != nullptr);
        return *self.m_chunks[slot];
    }

    template <typename Self>
    [[nodiscard]] decltype(auto) mesh_at(this Self&& self, u32 slot)
    {
        assert(slot < self.m_meshes.size());
        return (self.m_meshes[slot]);
    }

private:

    std::vector<std::unique_ptr<ChunkType>> m_chunks;
    std::vector<ChunkMesh> m_meshes;

    std::unique_ptr<u32[]> m_active_slots;
    std::unique_ptr<u32[]> m_slot_to_active;
    std::unique_ptr<u32[]> m_free_slots;

    u32 m_active_count = 0;
    u32 m_free_count = 0;
};

class LodPoolGroup
{
public:

    template <u32 Lod, typename Self>
        requires(Lod >= 1 && Lod <= MAX_LOD)
    [[nodiscard]] decltype(auto) pool(this Self&& self)
    {
        return std::get<Lod - 1u>(self.m_pools);
    }

    template <typename Self, typename Fn>
    decltype(auto) visit_lod(this Self&& self, u8 lod, Fn&& fn)
    {
        switch (lod)
        {
            case 1: return std::forward<Fn>(fn)(self.template pool<1>());
            case 2: return std::forward<Fn>(fn)(self.template pool<2>());
            case 3: return std::forward<Fn>(fn)(self.template pool<3>());
            case 4: return std::forward<Fn>(fn)(self.template pool<4>());
            default: return std::forward<Fn>(fn)(self.template pool<5>());
        }
    }

    template <typename Self, typename Fn>
    void for_each_pool(this Self&& self, Fn&& fn)
    {
        std::apply([&](auto&&... pools) { (fn(pools), ...); }, self.m_pools);
    }

    [[nodiscard]] u32 total_active_chunks() const
    {
        u32 total = 0;
        for_each_pool([&](const auto& p) { total += p.size(); });
        return total;
    }

    [[nodiscard]] static consteval u32 total_capacity()
    {
        return compute_lod_pool_capacity<1>() + compute_lod_pool_capacity<2>() + compute_lod_pool_capacity<3>() +
               compute_lod_pool_capacity<4>() + compute_lod_pool_capacity<5>();
    }

    template <typename Self, typename Fn>
    void for_each_active_mesh(this Self&& self, Fn&& fn)
    {
        self.for_each_pool([&](auto&& p) {
            const u32 count = p.size();
            for (u32 i = 0; i < count; ++i) fn(p.mesh_at(p.active_slot(i)));
        });
    }

    void destroy_all_gpu_resources(vulk::VulkanContext& ctx)
    {
        for_each_pool([&](auto& p) {
            const u32 count = p.size();
            for (u32 i = 0; i < count; ++i) p.mesh_at(p.active_slot(i)).destroy_gpu_resources(ctx);
        });
    }

private:

    std::tuple<ChunkPool<1>, ChunkPool<2>, ChunkPool<3>, ChunkPool<4>, ChunkPool<5>> m_pools {};
};

} // namespace vl
