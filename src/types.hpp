#pragma once

#include <cstddef>
#include <cstdint>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;

template <typename Tag = void>
struct Handle
{
    u32 index = ~0u;
    u32 generation = 0;

    constexpr Handle() = default;
    constexpr Handle(u32 idx, u32 gen) : index(idx), generation(gen) {}

    static constexpr Handle invalid() { return Handle {~0u, 0}; }

    constexpr bool is_valid() const { return index != ~0u; }
};