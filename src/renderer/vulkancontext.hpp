#pragma once

#include "types.hpp"

#include <array>
#include <span>
#include <string_view>

struct SDL_Window;

namespace vulk
{

using BufferHandle = Handle<struct BufferTag>;
using TextureHandle = Handle<struct TextureTag>;
using SamplerHandle = Handle<struct SamplerTag>;
using ShaderHandle = Handle<struct ShaderTag>;
using PipelineHandle = Handle<struct PipelineTag>;

enum class BufferUsage : u32
{
    None = 0,
    Vertex = 1u << 0,
    Index = 1u << 1,
    Uniform = 1u << 2,
    Storage = 1u << 3,
    Indirect = 1u << 4,
    TransferSrc = 1u << 5,
    TransferDst = 1u << 6,
    ShaderDeviceAddress = 1u << 7,
};

constexpr BufferUsage operator|(BufferUsage a, BufferUsage b)
{
    return static_cast<BufferUsage>(static_cast<u32>(a) | static_cast<u32>(b));
}

constexpr BufferUsage operator&(BufferUsage a, BufferUsage b)
{
    return static_cast<BufferUsage>(static_cast<u32>(a) & static_cast<u32>(b));
}

constexpr bool has_flag(BufferUsage flags, BufferUsage bit)
{
    return (static_cast<u32>(flags) & static_cast<u32>(bit)) != 0;
}

enum class MemoryUsage : u8
{
    GpuOnly = 0,
    CpuToGpu = 1,
    GpuToCpu = 2,
};

enum class TextureFormat : u8
{
    Undefined = 0,
    RGBA8_UNorm,
    RGBA8_sRGB,
    BGRA8_UNorm,
    BGRA8_sRGB,
    R8_UNorm,
    R16_Float,
    RGBA16_Float,
    R32_UInt,
    RGBA32_Float,
    D32_Float,
};

enum class TextureUsage : u32
{
    None = 0,
    Sampled = 1u << 0,
    Storage = 1u << 1,
    ColorAttachment = 1u << 2,
    DepthStencilAttachment = 1u << 3,
    TransferSrc = 1u << 4,
    TransferDst = 1u << 5,
};

constexpr TextureUsage operator|(TextureUsage a, TextureUsage b)
{
    return static_cast<TextureUsage>(static_cast<u32>(a) | static_cast<u32>(b));
}

constexpr TextureUsage operator&(TextureUsage a, TextureUsage b)
{
    return static_cast<TextureUsage>(static_cast<u32>(a) & static_cast<u32>(b));
}

constexpr bool has_flag(TextureUsage flags, TextureUsage bit)
{
    return (static_cast<u32>(flags) & static_cast<u32>(bit)) != 0;
}

enum class FilterMode : u8
{
    Nearest = 0,
    Linear = 1,
};

enum class AddressMode : u8
{
    Repeat = 0,
    ClampToEdge = 1,
    ClampToBorder = 2,
};

enum class VertexFormat : u8
{
    Float1 = 0,
    Float2,
    Float3,
    Float4,
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UNorm4x8,
};

enum class PrimitiveTopology : u8
{
    TriangleList = 0,
    TriangleStrip,
    LineList,
    PointList,
};

enum class PolygonMode : u8
{
    Fill = 0,
    Line = 1,
    Point = 2,
};

enum class CullMode : u8
{
    None = 0,
    Back = 1,
    Front = 2,
};

enum class FrontFace : u8
{
    CounterClockwise = 0,
    Clockwise = 1,
};

enum class CompareOp : u8
{
    Never = 0,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
};

struct ContextConfig
{
    std::string_view app_name = "";
    bool enable_validation = true;
    bool vsync = true;
    u32 frames_in_flight = 2;
    u64 staging_buffer_size = 64ull * 1024ull * 1024ull; // 64 MB per-frame upload capacity
};

struct RenderPassInfo
{
    std::array<f32, 4> clear_color = {0.06f, 0.08f, 0.12f, 1.0f};
    f32 clear_depth = 1.0f;
    bool should_clear_color = true;
    bool should_clear_depth = true;
};

struct VertexAttributeDesc
{
    u32 location = 0;
    u32 offset = 0;
    VertexFormat format = VertexFormat::Float3;
};

struct GraphicsPipelineDesc
{
    ShaderHandle vertex_shader {};
    ShaderHandle fragment_shader {};
    std::string_view vertex_entry = "vertexMain";
    std::string_view fragment_entry = "fragmentMain";

    // Leave attributes empty (vertex pulling)
    std::span<const VertexAttributeDesc> vertex_attributes {};
    u32 vertex_stride = 0;

    PrimitiveTopology topology = PrimitiveTopology::TriangleList;
    PolygonMode polygon_mode = PolygonMode::Fill;
    CullMode cull_mode = CullMode::Back;
    FrontFace front_face = FrontFace::CounterClockwise;

    bool depth_test = true;
    bool depth_write = true;
    CompareOp depth_compare = CompareOp::LessOrEqual;

    bool blend_enable = false;
    u32 push_constant_size = 128;
};

// Matches VkDrawIndirectCommand layout
struct DrawIndirectCommand
{
    u32 vertex_count = 0;
    u32 instance_count = 1;
    u32 first_vertex = 0;
    u32 first_instance = 0;
};

// Matches VkDrawIndexedIndirectCommand layout
struct DrawIndexedIndirectCommand
{
    u32 index_count = 0;
    u32 instance_count = 1;
    u32 first_index = 0;
    i32 vertex_offset = 0;
    u32 first_instance = 0;
};

// Matches VkDispatchIndirectCommand
struct DispatchIndirectCommand
{
    u32 group_count_x = 1;
    u32 group_count_y = 1;
    u32 group_count_z = 1;
};

struct GpuMemoryStats
{
    u64 allocated_bytes = 0;
    u64 used_bytes = 0;
    u32 active_buffers = 0;
    u32 active_textures = 0;
    u32 draw_calls_last_frame = 0;
    u32 triangles_last_frame = 0;
};

class VulkanContext
{
public:

    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&) = delete;
    VulkanContext& operator=(VulkanContext&&) = delete;

    // Context lifecycle
    bool init(SDL_Window* window, const ContextConfig& config = {});
    bool init();
    void shutdown();
    void wait_idle();

    bool on_resize(u32 width, u32 height);
    void set_vsync(bool enabled);

    [[nodiscard]] u32 get_render_width() const;
    [[nodiscard]] u32 get_render_height() const;
    [[nodiscard]] f32 get_aspect_ratio() const;
    [[nodiscard]] u32 get_current_frame_index() const;
    [[nodiscard]] GpuMemoryStats get_memory_stats() const;

    // Frame and dynamic rendering lifecycle
    bool begin_frame();
    void begin_rendering(const RenderPassInfo& pass_info = {});
    void end_rendering();
    void end_frame();

    // buffer management
    [[nodiscard]] BufferHandle
    create_buffer(u64 size_bytes, BufferUsage usage, MemoryUsage memory_usage, const void* initial_data = nullptr);
    void destroy_buffer(BufferHandle handle);

    bool upload_buffer_data(BufferHandle handle, u64 offset_bytes, std::span<const u8> data);
    bool upload_buffer_data(BufferHandle handle, u64 offset_bytes, const void* data, usize size_bytes);

    [[nodiscard]] void* map_buffer(BufferHandle handle);
    void unmap_buffer(BufferHandle handle);

    [[nodiscard]] u64 get_buffer_device_address(BufferHandle handle) const;
    [[nodiscard]] u64 get_buffer_size(BufferHandle handle) const;

    void copy_buffer(BufferHandle src, BufferHandle dst, u64 size_bytes, u64 src_offset = 0, u64 dst_offset = 0);
    void fill_buffer(BufferHandle dst, u32 value, u64 size_bytes, u64 offset_bytes = 0);

    [[nodiscard]] BufferHandle get_or_create_quad_index_buffer(u32 max_quads);

    // Texture and sampler management
    [[nodiscard]] TextureHandle create_texture_2d(
        u32 width, u32 height, TextureFormat format, TextureUsage usage, std::span<const u8> initial_data = {}, u32 mip_levels = 1
    );
    [[nodiscard]] TextureHandle create_texture_2d_array(
        u32 width, u32 height, u32 layer_count, TextureFormat format, TextureUsage usage, std::span<const u8> initial_data = {},
        u32 mip_levels = 1
    );
    [[nodiscard]] TextureHandle create_texture_3d(
        u32 width, u32 height, u32 depth, TextureFormat format, TextureUsage usage, std::span<const u8> initial_data = {}
    );
    bool upload_texture_layer(TextureHandle handle, u32 layer, std::span<const u8> data);
    void destroy_texture(TextureHandle handle);
    [[nodiscard]] u32 get_texture_bindless_index(TextureHandle handle) const;

    [[nodiscard]] SamplerHandle create_sampler(
        FilterMode min_filter = FilterMode::Nearest, FilterMode mag_filter = FilterMode::Nearest,
        AddressMode address_mode = AddressMode::Repeat, f32 max_anisotropy = 1.0f
    );
    void destroy_sampler(SamplerHandle handle);
    [[nodiscard]] u32 get_sampler_bindless_index(SamplerHandle handle) const;

    // Shader and pipeline management
    [[nodiscard]] ShaderHandle create_shader_from_file(std::string_view filepath);
    [[nodiscard]] ShaderHandle create_shader_from_spirv(std::span<const u32> spirv_words);
    void destroy_shader(ShaderHandle handle);

    [[nodiscard]] PipelineHandle create_graphics_pipeline(const GraphicsPipelineDesc& desc);
    [[nodiscard]] PipelineHandle create_compute_pipeline(
        ShaderHandle compute_shader, std::string_view entry_point = "computeMain", u32 push_constant_size = 128
    );
    void destroy_pipeline(PipelineHandle handle);

    // Command recording
    void bind_pipeline(PipelineHandle pipeline);
    void push_constants(const void* data, u32 size_bytes, u32 offset_bytes = 0);

    template <typename T>
    void push_constants(const T& data, u32 offset_bytes = 0)
    {
        push_constants(&data, static_cast<u32>(sizeof(T)), offset_bytes);
    }

    void bind_vertex_buffer(BufferHandle buffer, u64 offset_bytes = 0);
    void bind_index_buffer_u16(BufferHandle buffer, u64 offset_bytes = 0);
    void bind_index_buffer_u32(BufferHandle buffer, u64 offset_bytes = 0);

    void set_scissor(i32 x, i32 y, u32 width, u32 height);
    void set_viewport(f32 x, f32 y, f32 width, f32 height, f32 min_depth = 0.0f, f32 max_depth = 1.0f);

    // Direct draw commands
    void draw(u32 vertex_count, u32 instance_count = 1, u32 first_vertex = 0, u32 first_instance = 0);
    void
    draw_indexed(u32 index_count, u32 instance_count = 1, u32 first_index = 0, i32 vertex_offset = 0, u32 first_instance = 0);

    // Indirect commands
    void draw_indirect(
        BufferHandle indirect_buffer, u64 offset_bytes, u32 draw_count, u32 stride = static_cast<u32>(sizeof(DrawIndirectCommand))
    );
    void draw_indexed_indirect(
        BufferHandle indirect_buffer, u64 offset_bytes, u32 draw_count,
        u32 stride = static_cast<u32>(sizeof(DrawIndexedIndirectCommand))
    );
    void draw_indirect_count(
        BufferHandle indirect_buffer, u64 offset_bytes, BufferHandle count_buffer, u64 count_offset_bytes, u32 max_draw_count,
        u32 stride = static_cast<u32>(sizeof(DrawIndirectCommand))
    );
    void draw_indexed_indirect_count(
        BufferHandle indirect_buffer, u64 offset_bytes, BufferHandle count_buffer, u64 count_offset_bytes, u32 max_draw_count,
        u32 stride = static_cast<u32>(sizeof(DrawIndexedIndirectCommand))
    );

    // Compute dispatch and barriers
    void dispatch_compute(u32 group_count_x, u32 group_count_y = 1, u32 group_count_z = 1);
    void dispatch_compute_indirect(BufferHandle indirect_buffer, u64 offset_bytes = 0);

    void memory_barrier_compute_to_draw();
    void memory_barrier_compute_to_compute();
    void memory_barrier_transfer_to_compute();
    void memory_barrier_transfer_to_graphics();
};

} // namespace vulk