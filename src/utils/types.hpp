#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

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
    static constexpr u32 INVALID_INDEX = ~0u;

    u32 index = INVALID_INDEX;
    u32 generation = 0;

    constexpr Handle() = default;
    constexpr Handle(u32 idx, u32 gen) : index(idx), generation(gen) {}

    [[nodiscard]] static constexpr Handle invalid() { return Handle {INVALID_INDEX, 0}; }

    [[nodiscard]] constexpr bool is_valid() const { return index != INVALID_INDEX; }
};

