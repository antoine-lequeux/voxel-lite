#pragma once

#include "chunk.hpp"

#include <bit>
#include <cassert>
#include <limits>
#include <memory>
#include <vector>

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
    static constexpr u32 MAX_LOAD = (Capacity * 3u) / 4u; // 75% max load factor to guarantee low average probe length

    struct alignas(16) Slot
    {
        glm::ivec3 coord {0, 0, 0};
        u32 pool_index = INVALID_POOL_INDEX;
    };

    static_assert(sizeof(Slot) == 16, "Slot must be 16 bytes for cache-line alignment");

    FlatSpatialMap() : m_slots(std::make_unique<Slot[]>(Capacity)) { clear(); }

    [[nodiscard]] static constexpr u32 hash_coord(const glm::ivec3& v)
    {
        // Spatial prime mix and full Murmur3 32-bit avalanche bit mixing
        u32 h = (static_cast<u32>(v.x) * 73856093u) ^ (static_cast<u32>(v.y) * 19349663u) ^ (static_cast<u32>(v.z) * 83492791u);
        h ^= h >> 16u;
        h *= 0x85ebca6bu;
        h ^= h >> 13u;
        h *= 0xc2b2ae35u;
        h ^= h >> 16u;
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

    [[nodiscard]] bool contains(const glm::ivec3& coord) const { return find(coord) != INVALID_POOL_INDEX; }

    bool insert_or_assign(const glm::ivec3& coord, u32 pool_index)
    {
        assert(m_size < MAX_LOAD && "FlatSpatialMap capacity exceeded");
        if (m_size >= MAX_LOAD) return false;

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

    ChunkPool()
        : m_chunks(MaxChunks), m_meshes(std::make_unique<ChunkMesh[]>(MaxChunks)),
          m_active_slots(std::make_unique<u32[]>(MaxChunks)), m_slot_to_active(std::make_unique<u32[]>(MaxChunks)),
          m_free_slots(std::make_unique<u32[]>(MaxChunks))
    {
        for (u32 i = 0; i < MaxChunks; ++i)
        {
            m_slot_to_active[i] = INVALID_POOL_INDEX;
            // Push slots in reverse order so slot 0 is popped first
            m_free_slots[i] = (MaxChunks - 1u) - i;
        }
        m_free_count = MaxChunks;
    }

    [[nodiscard]] constexpr u32 size() const { return m_active_count; }
    [[nodiscard]] constexpr bool full() const { return m_free_count == 0u; }
    [[nodiscard]] static constexpr u32 capacity() { return MaxChunks; }

    [[nodiscard]] u32 active_slot(u32 active_index) const
    {
        assert(active_index < m_active_count);
        return m_active_slots[active_index];
    }

    // Allocates a stable chunk slot in-place
    [[nodiscard]] u32 allocate(const glm::ivec3& coord)
    {
        assert(m_free_count > 0u && "ChunkPool capacity exceeded");

        const u32 slot = m_free_slots[--m_free_count];

        // On-demand chunk allocation
        if (!m_chunks[slot]) m_chunks[slot] = std::make_unique<Chunk>();

        Chunk& chunk = *m_chunks[slot];
        chunk.coord = coord;
        chunk.mesh_dirty = true;

        ChunkMesh& mesh = m_meshes[slot];
        mesh.coord = coord;
        mesh.clear();

        // Register in dense active slots list
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

        // Destroy GPU resources and clear CPU vertex cache
        m_meshes[slot].destroy_gpu_resources(ctx);
        m_meshes[slot].clear();

        // Dense active list removal
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

    [[nodiscard]] Chunk& chunk_at(u32 slot) { return *m_chunks[slot]; }
    [[nodiscard]] const Chunk& chunk_at(u32 slot) const { return *m_chunks[slot]; }

    [[nodiscard]] ChunkMesh& mesh_at(u32 slot) { return m_meshes[slot]; }
    [[nodiscard]] const ChunkMesh& mesh_at(u32 slot) const { return m_meshes[slot]; }

    struct ActiveChunkRef
    {
        Chunk& chunk;
        ChunkMesh& mesh;
    };

    struct ConstActiveChunkRef
    {
        const Chunk& chunk;
        const ChunkMesh& mesh;
    };

    struct ActiveIterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = ActiveChunkRef;
        using difference_type = std::ptrdiff_t;

        ChunkPool* pool = nullptr;
        u32 index = 0;

        bool operator==(const ActiveIterator& other) const { return index == other.index; }
        bool operator!=(const ActiveIterator& other) const { return index != other.index; }

        ActiveIterator& operator++()
        {
            ++index;
            return *this;
        }
        ActiveIterator operator++(int)
        {
            auto tmp = *this;
            ++index;
            return tmp;
        }

        ActiveChunkRef operator*() const
        {
            const u32 slot = pool->m_active_slots[index];
            return {*pool->m_chunks[slot], pool->m_meshes[slot]};
        }
    };

    struct ConstActiveIterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = ConstActiveChunkRef;
        using difference_type = std::ptrdiff_t;

        const ChunkPool* pool = nullptr;
        u32 index = 0;

        bool operator==(const ConstActiveIterator& other) const { return index == other.index; }
        bool operator!=(const ConstActiveIterator& other) const { return index != other.index; }

        ConstActiveIterator& operator++()
        {
            ++index;
            return *this;
        }
        ConstActiveIterator operator++(int)
        {
            auto tmp = *this;
            ++index;
            return tmp;
        }

        ConstActiveChunkRef operator*() const
        {
            const u32 slot = pool->m_active_slots[index];
            return {*pool->m_chunks[slot], pool->m_meshes[slot]};
        }
    };

    struct ActiveRange
    {
        ChunkPool* pool = nullptr;
        ActiveIterator begin() const { return {pool, 0}; }
        ActiveIterator end() const { return {pool, pool->m_active_count}; }
    };

    struct ConstActiveRange
    {
        const ChunkPool* pool = nullptr;
        ConstActiveIterator begin() const { return {pool, 0}; }
        ConstActiveIterator end() const { return {pool, pool->m_active_count}; }
    };

    [[nodiscard]] ActiveRange zipped() { return {this}; }
    [[nodiscard]] ConstActiveRange zipped() const { return {this}; }

    struct ActiveMeshIterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = ChunkMesh;
        using difference_type = std::ptrdiff_t;

        ChunkPool* pool = nullptr;
        u32 index = 0;

        bool operator==(const ActiveMeshIterator& other) const { return index == other.index; }
        bool operator!=(const ActiveMeshIterator& other) const { return index != other.index; }

        ActiveMeshIterator& operator++()
        {
            ++index;
            return *this;
        }
        ActiveMeshIterator operator++(int)
        {
            auto tmp = *this;
            ++index;
            return tmp;
        }

        ChunkMesh& operator*() const
        {
            const u32 slot = pool->m_active_slots[index];
            return pool->m_meshes[slot];
        }
    };

    struct ConstActiveMeshIterator
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = const ChunkMesh;
        using difference_type = std::ptrdiff_t;

        const ChunkPool* pool = nullptr;
        u32 index = 0;

        bool operator==(const ConstActiveMeshIterator& other) const { return index == other.index; }
        bool operator!=(const ConstActiveMeshIterator& other) const { return index != other.index; }

        ConstActiveMeshIterator& operator++()
        {
            ++index;
            return *this;
        }
        ConstActiveMeshIterator operator++(int)
        {
            auto tmp = *this;
            ++index;
            return tmp;
        }

        const ChunkMesh& operator*() const
        {
            const u32 slot = pool->m_active_slots[index];
            return pool->m_meshes[slot];
        }
    };

    struct ActiveMeshRange
    {
        ChunkPool* pool = nullptr;
        ActiveMeshIterator begin() const { return {pool, 0}; }
        ActiveMeshIterator end() const { return {pool, pool->m_active_count}; }
    };

    struct ConstActiveMeshRange
    {
        const ChunkPool* pool = nullptr;
        ConstActiveMeshIterator begin() const { return {pool, 0}; }
        ConstActiveMeshIterator end() const { return {pool, pool->m_active_count}; }
    };

    [[nodiscard]] ActiveMeshRange active_meshes() { return {this}; }
    [[nodiscard]] ConstActiveMeshRange active_meshes() const { return {this}; }

private:

    std::vector<std::unique_ptr<Chunk>> m_chunks;
    std::unique_ptr<ChunkMesh[]> m_meshes;

    std::unique_ptr<u32[]> m_active_slots;
    std::unique_ptr<u32[]> m_slot_to_active;
    std::unique_ptr<u32[]> m_free_slots;

    u32 m_active_count = 0;
    u32 m_free_count = 0;
};

} // namespace vl