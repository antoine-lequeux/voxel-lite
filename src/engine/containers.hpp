#pragma once

#include "chunk.hpp"

#include <bit>
#include <cassert>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <utility>

#include <glm/glm.hpp>

namespace vl
{

inline constexpr u32 INVALID_POOL_INDEX = std::numeric_limits<u32>::max();

template <u32 Capacity = 4096>
    requires(std::has_single_bit(Capacity)) // Must be a power of 2 for bitwise modulo
class FlatSpatialMap
{
public:

    static constexpr u32 MASK = Capacity - 1u;

    struct alignas(16) Slot
    {
        glm::ivec3 coord {0, 0, 0};
        u32 pool_index = INVALID_POOL_INDEX;
    };

    static_assert(sizeof(Slot) == 16, "Slot must be 16 bytes for cache-line alignment");

    FlatSpatialMap() : m_slots(std::make_unique<Slot[]>(Capacity)) {}

    [[nodiscard]] static constexpr u32 hash_coord(const glm::ivec3& v)
    {
        // Spatial prime mix and Murmur3-style avalanche bit mixing
        u32 h = (static_cast<u32>(v.x) * 73856093u) ^ (static_cast<u32>(v.y) * 19349663u) ^ (static_cast<u32>(v.z) * 83492791u);
        h ^= h >> 16u;
        h *= 0x85ebca6bu;
        h ^= h >> 13u;
        return h;
    }

    [[nodiscard]] u32 find(const glm::ivec3& coord) const
    {
        u32 slot_idx = hash_coord(coord) & MASK;

        while (true)
        {
            const Slot& slot = m_slots[slot_idx];
            if (slot.pool_index == INVALID_POOL_INDEX) return INVALID_POOL_INDEX;
            if (slot.coord == coord) return slot.pool_index;

            slot_idx = (slot_idx + 1u) & MASK;
        }
    }

    bool insert_or_assign(const glm::ivec3& coord, u32 pool_index)
    {
        // Keep load factor <= 75% to guarantee low average probe length
        assert(m_size < (Capacity * 3u) / 4u && "FlatSpatialMap capacity exceeded");

        u32 slot_idx = hash_coord(coord) & MASK;

        while (true)
        {
            Slot& slot = m_slots[slot_idx];
            if (slot.pool_index == INVALID_POOL_INDEX)
            {
                slot.coord = coord;
                slot.pool_index = pool_index;
                ++m_size;
                return true; // Newly inserted
            }
            if (slot.coord == coord)
            {
                slot.pool_index = pool_index;
                return false; // Updated existing key
            }

            slot_idx = (slot_idx + 1u) & MASK;
        }
    }

    // Knuth algorithm R
    bool erase(const glm::ivec3& coord)
    {
        u32 i = hash_coord(coord) & MASK;

        while (true)
        {
            if (m_slots[i].pool_index == INVALID_POOL_INDEX) return false;
            if (m_slots[i].coord == coord) break;
            i = (i + 1u) & MASK;
        }

        u32 j = (i + 1u) & MASK;
        while (m_slots[j].pool_index != INVALID_POOL_INDEX)
        {
            const u32 k = hash_coord(m_slots[j].coord) & MASK;

            // Cyclically check if slot i lies on the probe path from k to j
            if (((j - i) & MASK) <= ((j - k) & MASK))
            {
                m_slots[i] = m_slots[j];
                i = j;
            }
            j = (j + 1u) & MASK;
        }

        m_slots[i].pool_index = INVALID_POOL_INDEX;
        --m_size;
        return true;
    }

    void clear()
    {
        for (u32 i = 0; i < Capacity; ++i) m_slots[i].pool_index = INVALID_POOL_INDEX;
        m_size = 0;
    }

    [[nodiscard]] constexpr u32 size() const { return m_size; }
    [[nodiscard]] static constexpr u32 capacity() { return Capacity; }

private:

    std::unique_ptr<Slot[]> m_slots;
    u32 m_size = 0;
};

template <u32 MaxChunks = 2048>
class ChunkPool
{
public:

    ChunkPool() : m_chunks(std::make_unique<Chunk[]>(MaxChunks)), m_meshes(std::make_unique<ChunkMesh[]>(MaxChunks)) {}

    [[nodiscard]] constexpr u32 size() const { return m_size; }
    [[nodiscard]] constexpr bool full() const { return m_size >= MaxChunks; }
    [[nodiscard]] static constexpr u32 capacity() { return MaxChunks; }

    // Allocates the next contiguous slot without copies
    [[nodiscard]] u32 allocate(const glm::ivec3& coord)
    {
        assert(m_size < MaxChunks && "ChunkPool capacity exceeded");

        const u32 idx = m_size++;
        Chunk& chunk = m_chunks[idx];
        chunk.coord = coord;
        chunk.mesh_dirty = true;

        m_meshes[idx].clear();
        return idx;
    }

    glm::ivec3 swap_and_pop(u32 removed_idx, vulk::VulkanContext& ctx)
    {
        assert(removed_idx < m_size);

        // Destroy the GPU buffer of the chunk being unloaded
        m_meshes[removed_idx].destroy_gpu_resources(ctx);
        m_meshes[removed_idx].clear();

        const u32 last_idx = --m_size;
        if (removed_idx != last_idx)
        {
            m_chunks[removed_idx] = m_chunks[last_idx];
            std::swap(m_meshes[removed_idx], m_meshes[last_idx]);
            return m_chunks[removed_idx].coord;
        }

        return {0, 0, 0};
    }

    [[nodiscard]] Chunk& chunk_at(u32 idx) { return m_chunks[idx]; }
    [[nodiscard]] const Chunk& chunk_at(u32 idx) const { return m_chunks[idx]; }

    [[nodiscard]] ChunkMesh& mesh_at(u32 idx) { return m_meshes[idx]; }
    [[nodiscard]] const ChunkMesh& mesh_at(u32 idx) const { return m_meshes[idx]; }

    [[nodiscard]] std::span<Chunk> chunks() { return {m_chunks.get(), m_size}; }
    [[nodiscard]] std::span<const Chunk> chunks() const { return {m_chunks.get(), m_size}; }

    [[nodiscard]] std::span<ChunkMesh> meshes() { return {m_meshes.get(), m_size}; }
    [[nodiscard]] std::span<const ChunkMesh> meshes() const { return {m_meshes.get(), m_size}; }

    [[nodiscard]] auto zipped() { return std::views::zip(chunks(), meshes()); }
    [[nodiscard]] auto zipped() const { return std::views::zip(chunks(), meshes()); }

private:

    std::unique_ptr<Chunk[]> m_chunks;
    std::unique_ptr<ChunkMesh[]> m_meshes;
    u32 m_size = 0;
};

} // namespace vl