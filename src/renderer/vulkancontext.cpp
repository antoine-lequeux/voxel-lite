#include "vulkancontext.hpp"

#include <volk.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wall"
    #pragma clang diagnostic ignored "-Wextra"
    #pragma clang diagnostic ignored "-Wpedantic"
    #pragma clang diagnostic ignored "-Wconversion"
    #pragma clang diagnostic ignored "-Wsign-conversion"
    #pragma clang diagnostic ignored "-Wold-style-cast"
    #pragma clang diagnostic ignored "-Wnull-dereference"
    #pragma clang diagnostic ignored "-Wshadow"
    #pragma clang diagnostic ignored "-Wunused-parameter"
    #pragma clang diagnostic ignored "-Wunused-variable"
    #pragma clang diagnostic ignored "-Wmissing-field-initializers"
    #pragma clang diagnostic ignored "-Wnullability-completeness"
    #pragma clang diagnostic ignored "-Wcast-align"
    #pragma clang diagnostic ignored "-Wimplicit-fallthrough"
    #pragma clang diagnostic ignored "-Wundef"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wall"
    #pragma GCC diagnostic ignored "-Wextra"
    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wconversion"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wold-style-cast"
    #pragma GCC diagnostic ignored "-Wnull-dereference"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #pragma GCC diagnostic ignored "-Wunused-variable"
    #pragma GCC diagnostic ignored "-Wmissing-field-initializers"
    #pragma GCC diagnostic ignored "-Wcast-align"
    #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
    #pragma GCC diagnostic ignored "-Wundef"
#elif defined(_MSC_VER)
    #pragma warning(push, 0)
#endif

#ifndef VK_BOOTSTRAP_NO_EXCEPTIONS
    #define VK_BOOTSTRAP_NO_EXCEPTIONS
#endif
#include <VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <fstream>
#include <print>

namespace vulk
{

namespace
{

// Slot-map pool for resource lookup
template <typename T, typename Tag>
class ResourcePool
{
public:

    struct Slot
    {
        T data {};
        u32 generation = 1;
        bool alive = false;
    };

    [[nodiscard]] Handle<Tag> insert(T&& item)
    {
        if (!m_free_indices.empty())
        {
            const u32 idx = m_free_indices.back();
            m_free_indices.pop_back();
            Slot& slot = m_slots[idx];
            slot.data = std::move(item);
            slot.alive = true;
            ++m_active_count;
            return Handle<Tag> {idx, slot.generation};
        }

        const auto idx = static_cast<u32>(m_slots.size());
        m_slots.push_back(Slot {.data = std::move(item), .generation = 1, .alive = true});
        ++m_active_count;
        return Handle<Tag> {idx, 1};
    }

    [[nodiscard]] T* get(Handle<Tag> handle)
    {
        if (!handle.is_valid() || handle.index >= m_slots.size()) return nullptr;
        Slot& slot = m_slots[handle.index];
        if (!slot.alive || slot.generation != handle.generation) return nullptr;
        return &slot.data;
    }

    [[nodiscard]] const T* get(Handle<Tag> handle) const
    {
        if (!handle.is_valid() || handle.index >= m_slots.size()) return nullptr;
        const Slot& slot = m_slots[handle.index];
        if (!slot.alive || slot.generation != handle.generation) return nullptr;
        return &slot.data;
    }

    bool extract(Handle<Tag> handle, T& out_item)
    {
        if (!handle.is_valid() || handle.index >= m_slots.size()) return false;
        Slot& slot = m_slots[handle.index];
        if (!slot.alive || slot.generation != handle.generation) return false;

        out_item = std::move(slot.data);
        slot.data = T {};
        slot.alive = false;
        ++slot.generation;
        m_free_indices.push_back(handle.index);
        --m_active_count;
        return true;
    }

    [[nodiscard]] u32 active_count() const { return m_active_count; }
    [[nodiscard]] std::span<Slot> slots() { return m_slots; }

    void clear()
    {
        m_slots.clear();
        m_free_indices.clear();
        m_active_count = 0;
    }

private:

    std::vector<Slot> m_slots;
    std::vector<u32> m_free_indices;
    u32 m_active_count = 0;
};

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    u64 size = 0;
    u64 device_address = 0;
    void* mapped_ptr = nullptr;
    BufferUsage usage = BufferUsage::None;
    MemoryUsage memory_usage = MemoryUsage::GpuOnly;
};

struct GpuTexture
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    u32 width = 0;
    u32 height = 0;
    u32 depth_or_layers = 1;
    u32 mip_levels = 1;
    u32 bindless_index = ~0u;
    bool is_3d = false;
    bool is_array = false;
};

struct GpuSampler
{
    VkSampler sampler = VK_NULL_HANDLE;
    u32 bindless_index = ~0u;
};

struct GpuShader
{
    VkShaderModule module = VK_NULL_HANDLE;
    std::string vertex_entry;
    std::string fragment_entry;
    std::string compute_entry;
};

struct GpuPipeline
{
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
};

struct FrameData
{
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore image_available_semaphore = VK_NULL_HANDLE;
    VkFence in_flight_fence = VK_NULL_HANDLE;

    GpuBuffer staging_buffer {};
    u64 staging_offset = 0;

    std::vector<GpuBuffer> pending_buffer_deletions;
    std::vector<GpuTexture> pending_texture_deletions;

    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    bool timestamp_query_active = false;
};

constexpr VkFormat to_vk_format(TextureFormat fmt)
{
    switch (fmt)
    {
        case TextureFormat::RGBA8_UNorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case TextureFormat::RGBA8_sRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case TextureFormat::BGRA8_UNorm: return VK_FORMAT_B8G8R8A8_UNORM;
        case TextureFormat::BGRA8_sRGB: return VK_FORMAT_B8G8R8A8_SRGB;
        case TextureFormat::R8_UNorm: return VK_FORMAT_R8_UNORM;
        case TextureFormat::R16_Float: return VK_FORMAT_R16_SFLOAT;
        case TextureFormat::RGBA16_Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case TextureFormat::R32_UInt: return VK_FORMAT_R32_UINT;
        case TextureFormat::RGBA32_Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case TextureFormat::D32_Float: return VK_FORMAT_D32_SFLOAT;
        case TextureFormat::Undefined:
        default: return VK_FORMAT_UNDEFINED;
    }
}

constexpr VkFormat to_vk_vertex_format(VertexFormat fmt)
{
    switch (fmt)
    {
        case VertexFormat::Float1: return VK_FORMAT_R32_SFLOAT;
        case VertexFormat::Float2: return VK_FORMAT_R32G32_SFLOAT;
        case VertexFormat::Float3: return VK_FORMAT_R32G32B32_SFLOAT;
        case VertexFormat::Float4: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case VertexFormat::UInt1: return VK_FORMAT_R32_UINT;
        case VertexFormat::UInt2: return VK_FORMAT_R32G32_UINT;
        case VertexFormat::UInt3: return VK_FORMAT_R32G32B32_UINT;
        case VertexFormat::UInt4: return VK_FORMAT_R32G32B32A32_UINT;
        case VertexFormat::UNorm4x8: return VK_FORMAT_R8G8B8A8_UNORM;
        default: return VK_FORMAT_R32G32B32_SFLOAT;
    }
}

constexpr VkPrimitiveTopology to_vk_topology(PrimitiveTopology topo)
{
    switch (topo)
    {
        case PrimitiveTopology::TriangleList: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        case PrimitiveTopology::TriangleStrip: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case PrimitiveTopology::LineList: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case PrimitiveTopology::PointList: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

constexpr VkPolygonMode to_vk_polygon_mode(PolygonMode mode)
{
    switch (mode)
    {
        case PolygonMode::Fill: return VK_POLYGON_MODE_FILL;
        case PolygonMode::Line: return VK_POLYGON_MODE_LINE;
        case PolygonMode::Point: return VK_POLYGON_MODE_POINT;
        default: return VK_POLYGON_MODE_FILL;
    }
}

constexpr VkCullModeFlags to_vk_cull_mode(CullMode mode)
{
    switch (mode)
    {
        case CullMode::None: return VK_CULL_MODE_NONE;
        case CullMode::Back: return VK_CULL_MODE_BACK_BIT;
        case CullMode::Front: return VK_CULL_MODE_FRONT_BIT;
        default: return VK_CULL_MODE_BACK_BIT;
    }
}

constexpr VkFrontFace to_vk_front_face(FrontFace face)
{
    return (face == FrontFace::CounterClockwise) ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
}

constexpr VkCompareOp to_vk_compare_op(CompareOp op)
{
    switch (op)
    {
        case CompareOp::Never: return VK_COMPARE_OP_NEVER;
        case CompareOp::Less: return VK_COMPARE_OP_LESS;
        case CompareOp::Equal: return VK_COMPARE_OP_EQUAL;
        case CompareOp::LessOrEqual: return VK_COMPARE_OP_LESS_OR_EQUAL;
        case CompareOp::Greater: return VK_COMPARE_OP_GREATER;
        case CompareOp::NotEqual: return VK_COMPARE_OP_NOT_EQUAL;
        case CompareOp::GreaterOrEqual: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case CompareOp::Always: return VK_COMPARE_OP_ALWAYS;
        default: return VK_COMPARE_OP_LESS_OR_EQUAL;
    }
}

// Parse SPIR-V header instructions to extract entry point names
void parse_spirv_entry_points(std::span<const u32> words, GpuShader& out_shader)
{
    constexpr u32 SPIRV_MAGIC = 0x07230203u;
    constexpr u32 OP_ENTRY_POINT = 15u;

    if (words.size() < 5 || words[0] != SPIRV_MAGIC) return;

    usize idx = 5;
    while (idx < words.size())
    {
        const u32 instruction = words[idx];
        const u32 word_count = instruction >> 16u;
        const u32 opcode = instruction & 0xFFFFu;

        if (word_count == 0 || idx + word_count > words.size()) break;

        if (opcode == OP_ENTRY_POINT && word_count >= 4)
        {
            const u32 exec_model = words[idx + 1];
            const char* name_ptr = reinterpret_cast<const char*>(&words[idx + 3]);
            const usize max_chars = (word_count - 3) * sizeof(u32);
            const usize name_len = strnlen(name_ptr, max_chars);
            std::string entry_name(name_ptr, name_len);

            if (exec_model == 0) // Vertex
                out_shader.vertex_entry = std::move(entry_name);
            else if (exec_model == 4) // Fragment
                out_shader.fragment_entry = std::move(entry_name);
            else if (exec_model == 5) // GLCompute
                out_shader.compute_entry = std::move(entry_name);
        }

        idx += word_count;
    }
}

} // namespace

struct VulkanContext::Impl
{
    SDL_Window* window = nullptr;
    ContextConfig config {};
    bool initialized = false;
    bool headless = false;

    vkb::Instance vkb_instance {};
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    vkb::PhysicalDevice vkb_physical_device {};
    vkb::Device vkb_device {};
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphics_queue = VK_NULL_HANDLE;
    u32 graphics_queue_family = 0;

    VmaAllocator allocator = VK_NULL_HANDLE;

    vkb::Swapchain vkb_swapchain {};
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;
    VkFormat depth_format = VK_FORMAT_D32_SFLOAT;
    u32 render_width = 0;
    u32 render_height = 0;
    std::vector<VkImage> swapchain_images;
    std::vector<VkImageView> swapchain_image_views;
    std::vector<VkSemaphore> render_finished_semaphores;
    GpuTexture depth_attachment {};
    bool swapchain_needs_rebuild = false;

    std::vector<FrameData> frames;
    u32 current_frame_index = 0;
    u32 current_swapchain_image = 0;
    bool frame_active = false;
    bool rendering_active = false;

    VkCommandPool immediate_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer immediate_cmd = VK_NULL_HANDLE;
    VkFence immediate_fence = VK_NULL_HANDLE;

    VkDescriptorSetLayout bindless_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool bindless_pool = VK_NULL_HANDLE;
    VkDescriptorSet bindless_set = VK_NULL_HANDLE;
    VkPipelineLayout global_pipeline_layout = VK_NULL_HANDLE;
    VkPipelineBindPoint active_bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

    ResourcePool<GpuBuffer, BufferTag> buffers;
    ResourcePool<GpuTexture, TextureTag> textures;
    ResourcePool<GpuSampler, SamplerTag> samplers;
    ResourcePool<GpuShader, ShaderTag> shaders;
    ResourcePool<GpuPipeline, PipelineTag> pipelines;

    BufferHandle quad_index_buffer {};
    u32 quad_index_capacity = 0;

    u32 draw_calls_current_frame = 0;
    u32 triangles_current_frame = 0;
    u32 draw_calls_last_frame = 0;
    u32 triangles_last_frame = 0;

    f32 timestamp_period = 1.0f;
    f32 gpu_render_time_ms = 0.0f;

    bool allocate_raw_buffer(u64 size_bytes, BufferUsage usage, MemoryUsage mem_usage, GpuBuffer& out_buf)
    {
        VkBufferUsageFlags vk_usage = 0;
        if (has_flag(usage, BufferUsage::Vertex)) vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (has_flag(usage, BufferUsage::Index)) vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (has_flag(usage, BufferUsage::Uniform)) vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (has_flag(usage, BufferUsage::Storage)) vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (has_flag(usage, BufferUsage::Indirect)) vk_usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (has_flag(usage, BufferUsage::TransferSrc)) vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (has_flag(usage, BufferUsage::TransferDst)) vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (has_flag(usage, BufferUsage::ShaderDeviceAddress)) vk_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        if (mem_usage == MemoryUsage::GpuOnly)
            vk_usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        VkBufferCreateInfo buf_info {};
        buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buf_info.size = size_bytes;
        buf_info.usage = vk_usage;
        buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;

        if (mem_usage == MemoryUsage::CpuToGpu)
            alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        else if (mem_usage == MemoryUsage::GpuToCpu)
            alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        else
            alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        VmaAllocationInfo result_info {};
        if (vmaCreateBuffer(allocator, &buf_info, &alloc_info, &out_buf.buffer, &out_buf.allocation, &result_info) != VK_SUCCESS)
        {
            std::println(stderr, "[VulkanContext] Failed to allocate buffer of size {} bytes", size_bytes);
            return false;
        }

        out_buf.size = size_bytes;
        out_buf.mapped_ptr = result_info.pMappedData;
        out_buf.usage = usage;
        out_buf.memory_usage = mem_usage;

        if ((vk_usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0)
        {
            VkBufferDeviceAddressInfo bda_info {};
            bda_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            bda_info.buffer = out_buf.buffer;
            out_buf.device_address = vkGetBufferDeviceAddress(device, &bda_info);
        }

        return true;
    }

    void destroy_raw_buffer(GpuBuffer& buf)
    {
        if (buf.buffer != VK_NULL_HANDLE && allocator != VK_NULL_HANDLE) vmaDestroyBuffer(allocator, buf.buffer, buf.allocation);
        buf = GpuBuffer {};
    }

    void destroy_raw_texture(GpuTexture& tex)
    {
        if (tex.view != VK_NULL_HANDLE && device != VK_NULL_HANDLE) vkDestroyImageView(device, tex.view, nullptr);
        if (tex.image != VK_NULL_HANDLE && allocator != VK_NULL_HANDLE) vmaDestroyImage(allocator, tex.image, tex.allocation);
        tex = GpuTexture {};
    }

    template <typename Fn>
    bool immediate_submit(Fn&& record_fn)
    {
        if (vkResetCommandPool(device, immediate_cmd_pool, 0) != VK_SUCCESS) return false;

        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(immediate_cmd, &begin_info) != VK_SUCCESS) return false;

        record_fn(immediate_cmd);

        if (vkEndCommandBuffer(immediate_cmd) != VK_SUCCESS) return false;

        VkCommandBufferSubmitInfo cmd_submit {};
        cmd_submit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmd_submit.commandBuffer = immediate_cmd;

        VkSubmitInfo2 submit_info {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submit_info.commandBufferInfoCount = 1;
        submit_info.pCommandBufferInfos = &cmd_submit;

        if (vkResetFences(device, 1, &immediate_fence) != VK_SUCCESS) return false;

        if (vkQueueSubmit2(graphics_queue, 1, &submit_info, immediate_fence) != VK_SUCCESS) return false;

        return vkWaitForFences(device, 1, &immediate_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    }

    bool create_depth_attachment(u32 width, u32 height)
    {
        destroy_raw_texture(depth_attachment);

        VkImageCreateInfo img_info {};
        img_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        img_info.imageType = VK_IMAGE_TYPE_2D;
        img_info.format = depth_format;
        img_info.extent = {width, height, 1};
        img_info.mipLevels = 1;
        img_info.arrayLayers = 1;
        img_info.samples = VK_SAMPLE_COUNT_1_BIT;
        img_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        img_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo alloc_info {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

        if (vmaCreateImage(allocator, &img_info, &alloc_info, &depth_attachment.image, &depth_attachment.allocation, nullptr) !=
            VK_SUCCESS)
        {
            std::println(stderr, "[VulkanContext] Failed to create depth attachment image");
            return false;
        }

        VkImageViewCreateInfo view_info {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = depth_attachment.image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = depth_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &view_info, nullptr, &depth_attachment.view) != VK_SUCCESS)
        {
            std::println(stderr, "[VulkanContext] Failed to create depth attachment view");
            return false;
        }

        depth_attachment.vk_format = depth_format;
        depth_attachment.width = width;
        depth_attachment.height = height;
        return true;
    }

    void destroy_swapchain_resources()
    {
        for (VkSemaphore sem : render_finished_semaphores)
            if (sem != VK_NULL_HANDLE) vkDestroySemaphore(device, sem, nullptr);
        render_finished_semaphores.clear();

        destroy_raw_texture(depth_attachment);

        if (!swapchain_image_views.empty())
        {
            vkb_swapchain.destroy_image_views(swapchain_image_views);
            swapchain_image_views.clear();
        }
        swapchain_images.clear();

        if (vkb_swapchain.swapchain != VK_NULL_HANDLE)
        {
            vkb::destroy_swapchain(vkb_swapchain);
            vkb_swapchain.swapchain = VK_NULL_HANDLE;
        }
    }

    bool build_swapchain(u32 requested_width, u32 requested_height)
    {
        if (headless || window == nullptr) return true;

        i32 win_w = static_cast<i32>(requested_width);
        i32 win_h = static_cast<i32>(requested_height);
        if (win_w == 0 || win_h == 0) SDL_GetWindowSizeInPixels(window, &win_w, &win_h);

        if (win_w <= 0 || win_h <= 0) return false;

        vkDeviceWaitIdle(device);

        for (VkSemaphore sem : render_finished_semaphores)
            if (sem != VK_NULL_HANDLE) vkDestroySemaphore(device, sem, nullptr);
        render_finished_semaphores.clear();

        if (!swapchain_image_views.empty())
        {
            vkb_swapchain.destroy_image_views(swapchain_image_views);
            swapchain_image_views.clear();
        }

        vkb::SwapchainBuilder builder {vkb_device};
        builder.set_old_swapchain(vkb_swapchain)
            .set_desired_extent(static_cast<u32>(win_w), static_cast<u32>(win_h))
            .set_desired_format(VkSurfaceFormatKHR {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .add_fallback_format(VkSurfaceFormatKHR {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
            .add_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        if (config.vsync)
        {
            builder.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR);
        }
        else
        {
            builder.set_desired_present_mode(VK_PRESENT_MODE_IMMEDIATE_KHR)
                .add_fallback_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
                .add_fallback_present_mode(VK_PRESENT_MODE_FIFO_KHR);
        }

        auto swap_ret = builder.build();
        if (!swap_ret)
        {
            std::println(stderr, "[VulkanContext] Failed to build swapchain: {}", swap_ret.error().message());
            return false;
        }

        if (vkb_swapchain.swapchain != VK_NULL_HANDLE) vkb::destroy_swapchain(vkb_swapchain);

        vkb_swapchain = swap_ret.value();
        swapchain_format = vkb_swapchain.image_format;
        render_width = vkb_swapchain.extent.width;
        render_height = vkb_swapchain.extent.height;

        auto images_ret = vkb_swapchain.get_images();
        auto views_ret = vkb_swapchain.get_image_views();
        if (!images_ret || !views_ret)
        {
            std::println(stderr, "[VulkanContext] Failed to get swapchain images/views");
            return false;
        }

        swapchain_images = images_ret.value();
        swapchain_image_views = views_ret.value();

        render_finished_semaphores.resize(swapchain_images.size(), VK_NULL_HANDLE);
        VkSemaphoreCreateInfo sem_info {};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (VkSemaphore& sem : render_finished_semaphores)
            if (vkCreateSemaphore(device, &sem_info, nullptr, &sem) != VK_SUCCESS) return false;

        if (!create_depth_attachment(render_width, render_height)) return false;

        swapchain_needs_rebuild = false;
        return true;
    }

    bool init_bindless_descriptors()
    {
        constexpr u32 MAX_BINDLESS_TEXTURES = 1024;
        constexpr u32 MAX_BINDLESS_SAMPLERS = 64;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings {};

        // Binding 0: sampled textures
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[0].descriptorCount = MAX_BINDLESS_TEXTURES;
        bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

        // Binding 1: samplers
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        bindings[1].descriptorCount = MAX_BINDLESS_SAMPLERS;
        bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

        constexpr VkDescriptorBindingFlags bind_flags =
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        std::array<VkDescriptorBindingFlags, 2> binding_flags {bind_flags, bind_flags};

        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
        flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flags_info.bindingCount = static_cast<u32>(binding_flags.size());
        flags_info.pBindingFlags = binding_flags.data();

        VkDescriptorSetLayoutCreateInfo layout_info {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.pNext = &flags_info;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layout_info.bindingCount = static_cast<u32>(bindings.size());
        layout_info.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &bindless_set_layout) != VK_SUCCESS) return false;

        std::array<VkDescriptorPoolSize, 2> pool_sizes {};
        pool_sizes[0] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MAX_BINDLESS_TEXTURES};
        pool_sizes[1] = {VK_DESCRIPTOR_TYPE_SAMPLER, MAX_BINDLESS_SAMPLERS};

        VkDescriptorPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = static_cast<u32>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();

        if (vkCreateDescriptorPool(device, &pool_info, nullptr, &bindless_pool) != VK_SUCCESS) return false;

        VkDescriptorSetAllocateInfo alloc_info {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = bindless_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &bindless_set_layout;

        if (vkAllocateDescriptorSets(device, &alloc_info, &bindless_set) != VK_SUCCESS) return false;

        VkPushConstantRange pc_range {};
        pc_range.stageFlags = VK_SHADER_STAGE_ALL;
        pc_range.offset = 0;
        pc_range.size = 128;

        VkPipelineLayoutCreateInfo pipe_layout_info {};
        pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipe_layout_info.setLayoutCount = 1;
        pipe_layout_info.pSetLayouts = &bindless_set_layout;
        pipe_layout_info.pushConstantRangeCount = 1;
        pipe_layout_info.pPushConstantRanges = &pc_range;

        return vkCreatePipelineLayout(device, &pipe_layout_info, nullptr, &global_pipeline_layout) == VK_SUCCESS;
    }

    void flush_frame_deletions(FrameData& frame)
    {
        for (GpuBuffer& buf : frame.pending_buffer_deletions) destroy_raw_buffer(buf);
        frame.pending_buffer_deletions.clear();

        for (GpuTexture& tex : frame.pending_texture_deletions) destroy_raw_texture(tex);
        frame.pending_texture_deletions.clear();
    }
};

VulkanContext::~VulkanContext()
{
    shutdown();
}

bool VulkanContext::init()
{
    return init(nullptr, ContextConfig {});
}

bool VulkanContext::init(SDL_Window* window, const ContextConfig& config)
{
    if (m_impl != nullptr) return true;

    if (volkInitialize() != VK_SUCCESS)
    {
        std::println(stderr, "[VulkanContext] Failed to initialize Volk (Vulkan loader not found)");
        return false;
    }

    m_impl = new Impl();
    m_impl->window = window;
    m_impl->config = config;
    m_impl->headless = (window == nullptr);

    vkb::InstanceBuilder inst_builder;
    inst_builder.set_app_name(config.app_name.data())
        .set_app_version(config.app_version_major, config.app_version_minor, config.app_version_patch)
        .require_api_version(1, 3, 0);

    if (config.enable_validation)
    {
        inst_builder.request_validation_layers().use_default_debug_messenger().set_debug_messenger_type(
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
        );
    }

    if (window != nullptr)
    {
        u32 ext_count = 0;
        const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);
        if (sdl_exts == nullptr)
        {
            std::println(stderr, "[VulkanContext] SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError());
            shutdown();
            return false;
        }
        for (u32 i = 0; i < ext_count; ++i) inst_builder.enable_extension(sdl_exts[i]);
    }
    else
    {
        inst_builder.set_headless(true);
    }

    auto inst_ret = inst_builder.build();
    if (!inst_ret)
    {
        std::println(stderr, "[VulkanContext] Failed to create Vulkan instance: {}", inst_ret.error().message());
        shutdown();
        return false;
    }

    m_impl->vkb_instance = inst_ret.value();
    volkLoadInstance(m_impl->vkb_instance.instance);

    if (window != nullptr)
    {
        if (!SDL_Vulkan_CreateSurface(window, m_impl->vkb_instance.instance, nullptr, &m_impl->surface))
        {
            std::println(stderr, "[VulkanContext] Failed to create SDL3 Vulkan surface: {}", SDL_GetError());
            shutdown();
            return false;
        }
    }

    VkPhysicalDeviceVulkan13Features features13 {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12 {};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.scalarBlockLayout = VK_TRUE;
    features12.drawIndirectCount = VK_TRUE;

    VkPhysicalDeviceFeatures features10 {};
    features10.multiDrawIndirect = VK_TRUE;
    features10.drawIndirectFirstInstance = VK_TRUE;
    features10.fillModeNonSolid = VK_TRUE;
    features10.samplerAnisotropy = VK_TRUE;

    vkb::PhysicalDeviceSelector selector {m_impl->vkb_instance};
    selector.set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_required_features(features10);

    if (m_impl->surface != VK_NULL_HANDLE) selector.set_surface(m_impl->surface);

    auto phys_ret = selector.select();
    if (!phys_ret)
    {
        std::println(stderr, "[VulkanContext] Failed to select suitable GPU: {}", phys_ret.error().message());
        shutdown();
        return false;
    }

    m_impl->vkb_physical_device = phys_ret.value();
    std::println("[VulkanContext] Selected GPU: {}", m_impl->vkb_physical_device.name);

    vkb::DeviceBuilder dev_builder {m_impl->vkb_physical_device};
    auto dev_ret = dev_builder.build();
    if (!dev_ret)
    {
        std::println(stderr, "[VulkanContext] Failed to create logical device: {}", dev_ret.error().message());
        shutdown();
        return false;
    }

    m_impl->vkb_device = dev_ret.value();
    m_impl->device = m_impl->vkb_device.device;
    volkLoadDevice(m_impl->device);

    m_impl->timestamp_period = m_impl->vkb_physical_device.properties.limits.timestampPeriod;
    if (m_impl->timestamp_period <= 0.0f) m_impl->timestamp_period = 1.0f;

    auto queue_ret = m_impl->vkb_device.get_queue(vkb::QueueType::graphics);
    auto queue_fam_ret = m_impl->vkb_device.get_queue_index(vkb::QueueType::graphics);
    if (!queue_ret || !queue_fam_ret)
    {
        std::println(stderr, "[VulkanContext] Failed to retrieve graphics queue");
        shutdown();
        return false;
    }

    m_impl->graphics_queue = queue_ret.value();
    m_impl->graphics_queue_family = queue_fam_ret.value();

    VmaVulkanFunctions vma_funcs {};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info {};
    alloc_info.physicalDevice = m_impl->vkb_physical_device.physical_device;
    alloc_info.device = m_impl->device;
    alloc_info.instance = m_impl->vkb_instance.instance;
    alloc_info.vulkanApiVersion = VK_API_VERSION_1_3;
    alloc_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    alloc_info.pVulkanFunctions = &vma_funcs;

    if (vmaCreateAllocator(&alloc_info, &m_impl->allocator) != VK_SUCCESS)
    {
        std::println(stderr, "[VulkanContext] Failed to initialize Vulkan Memory Allocator (VMA)");
        shutdown();
        return false;
    }

    VkCommandPoolCreateInfo imm_pool_info {};
    imm_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    imm_pool_info.queueFamilyIndex = m_impl->graphics_queue_family;
    imm_pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(m_impl->device, &imm_pool_info, nullptr, &m_impl->immediate_cmd_pool) != VK_SUCCESS)
    {
        std::println(stderr, "[VulkanContext] Failed to create a command pool");
        shutdown();
        return false;
    }

    VkCommandBufferAllocateInfo imm_cmd_info {};
    imm_cmd_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    imm_cmd_info.commandPool = m_impl->immediate_cmd_pool;
    imm_cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    imm_cmd_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(m_impl->device, &imm_cmd_info, &m_impl->immediate_cmd) != VK_SUCCESS)
    {
        shutdown();
        return false;
    }

    VkFenceCreateInfo fence_info {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(m_impl->device, &fence_info, nullptr, &m_impl->immediate_fence) != VK_SUCCESS)
    {
        shutdown();
        return false;
    }

    const u32 frame_count = std::max(1u, config.frames_in_flight);
    m_impl->frames.resize(frame_count);

    VkSemaphoreCreateInfo sem_info {};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (FrameData& frame : m_impl->frames)
    {
        VkCommandPoolCreateInfo pool_info {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = m_impl->graphics_queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(m_impl->device, &pool_info, nullptr, &frame.cmd_pool) != VK_SUCCESS)
        {
            shutdown();
            return false;
        }

        VkCommandBufferAllocateInfo cmd_alloc {};
        cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmd_alloc.commandPool = frame.cmd_pool;
        cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmd_alloc.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(m_impl->device, &cmd_alloc, &frame.cmd) != VK_SUCCESS)
        {
            shutdown();
            return false;
        }

        if (vkCreateSemaphore(m_impl->device, &sem_info, nullptr, &frame.image_available_semaphore) != VK_SUCCESS ||
            vkCreateFence(m_impl->device, &fence_info, nullptr, &frame.in_flight_fence) != VK_SUCCESS)
        {
            shutdown();
            return false;
        }

        VkQueryPoolCreateInfo query_pool_info {};
        query_pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_info.queryCount = 2;
        if (vkCreateQueryPool(m_impl->device, &query_pool_info, nullptr, &frame.timestamp_query_pool) != VK_SUCCESS)
            frame.timestamp_query_pool = VK_NULL_HANDLE;

        if (config.staging_buffer_size > 0)
        {
            const u64 per_frame_staging = std::max<u64>(4ull * 1024ull * 1024ull, config.staging_buffer_size / frame_count);
            if (!m_impl->allocate_raw_buffer(
                    per_frame_staging, BufferUsage::TransferSrc, MemoryUsage::CpuToGpu, frame.staging_buffer
                ))
            {
                shutdown();
                return false;
            }
        }
    }

    if (!m_impl->init_bindless_descriptors())
    {
        std::println(stderr, "[VulkanContext] Failed to initialize bindless descriptor layout");
        shutdown();
        return false;
    }

    if (!m_impl->headless && !m_impl->build_swapchain(0, 0))
    {
        shutdown();
        return false;
    }

    m_impl->initialized = true;
    return true;
}

void VulkanContext::shutdown()
{
    if (m_impl == nullptr) return;

    shutdown_imgui();

    if (m_impl->device != VK_NULL_HANDLE) vkDeviceWaitIdle(m_impl->device);

    // Destroy cached global quad index buffer
    if (m_impl->quad_index_buffer.is_valid())
    {
        destroy_buffer(m_impl->quad_index_buffer);
        m_impl->quad_index_buffer = {};
    }

    // Destroy all remaining pipelines, shaders, samplers, textures, and buffers
    for (auto& slot : m_impl->pipelines.slots())
        if (slot.alive && slot.data.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_impl->device, slot.data.pipeline, nullptr);
    m_impl->pipelines.clear();

    for (auto& slot : m_impl->shaders.slots())
        if (slot.alive && slot.data.module != VK_NULL_HANDLE) vkDestroyShaderModule(m_impl->device, slot.data.module, nullptr);
    m_impl->shaders.clear();

    for (auto& slot : m_impl->samplers.slots())
        if (slot.alive && slot.data.sampler != VK_NULL_HANDLE) vkDestroySampler(m_impl->device, slot.data.sampler, nullptr);
    m_impl->samplers.clear();

    for (auto& slot : m_impl->textures.slots())
        if (slot.alive) m_impl->destroy_raw_texture(slot.data);
    m_impl->textures.clear();

    for (auto& slot : m_impl->buffers.slots())
        if (slot.alive) m_impl->destroy_raw_buffer(slot.data);
    m_impl->buffers.clear();

    // Clean up per-frame resources
    for (FrameData& frame : m_impl->frames)
    {
        m_impl->flush_frame_deletions(frame);
        m_impl->destroy_raw_buffer(frame.staging_buffer);

        if (frame.timestamp_query_pool != VK_NULL_HANDLE)
        {
            vkDestroyQueryPool(m_impl->device, frame.timestamp_query_pool, nullptr);
            frame.timestamp_query_pool = VK_NULL_HANDLE;
        }

        if (frame.image_available_semaphore != VK_NULL_HANDLE)
            vkDestroySemaphore(m_impl->device, frame.image_available_semaphore, nullptr);
        if (frame.in_flight_fence != VK_NULL_HANDLE) vkDestroyFence(m_impl->device, frame.in_flight_fence, nullptr);
        if (frame.cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(m_impl->device, frame.cmd_pool, nullptr);
    }
    m_impl->frames.clear();

    // Clean up immediate command context
    if (m_impl->immediate_fence != VK_NULL_HANDLE) vkDestroyFence(m_impl->device, m_impl->immediate_fence, nullptr);
    if (m_impl->immediate_cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(m_impl->device, m_impl->immediate_cmd_pool, nullptr);

    // Clean up bindless descriptors and global layout
    if (m_impl->global_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(m_impl->device, m_impl->global_pipeline_layout, nullptr);
    if (m_impl->bindless_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_impl->device, m_impl->bindless_pool, nullptr);
    if (m_impl->bindless_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(m_impl->device, m_impl->bindless_set_layout, nullptr);

    // Clean up swapchain and depth attachment
    m_impl->destroy_swapchain_resources();

    // Destroy VMA allocator
    if (m_impl->allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(m_impl->allocator);
        m_impl->allocator = VK_NULL_HANDLE;
    }

    // Destroy logical device, surface, and instance
    if (m_impl->device != VK_NULL_HANDLE)
    {
        vkb::destroy_device(m_impl->vkb_device);
        m_impl->device = VK_NULL_HANDLE;
    }

    if (m_impl->surface != VK_NULL_HANDLE && m_impl->vkb_instance.instance != VK_NULL_HANDLE)
    {
        SDL_Vulkan_DestroySurface(m_impl->vkb_instance.instance, m_impl->surface, nullptr);
        m_impl->surface = VK_NULL_HANDLE;
    }

    if (m_impl->vkb_instance.instance != VK_NULL_HANDLE)
    {
        vkb::destroy_instance(m_impl->vkb_instance);
        m_impl->vkb_instance.instance = VK_NULL_HANDLE;
    }

    delete m_impl;
    m_impl = nullptr;
}

void VulkanContext::wait_idle()
{
    if (m_impl != nullptr && m_impl->device != VK_NULL_HANDLE) vkDeviceWaitIdle(m_impl->device);
}

bool VulkanContext::on_resize(u32 width, u32 height)
{
    if (m_impl == nullptr || m_impl->headless) return false;

    if (width == 0 || height == 0)
    {
        m_impl->swapchain_needs_rebuild = true;
        return false;
    }

    return m_impl->build_swapchain(width, height);
}

void VulkanContext::set_vsync(bool enabled)
{
    if (m_impl == nullptr) return;
    if (m_impl->config.vsync != enabled)
    {
        m_impl->config.vsync = enabled;
        m_impl->swapchain_needs_rebuild = true;
    }
}

bool VulkanContext::is_vsync_enabled() const
{
    return (m_impl != nullptr) ? m_impl->config.vsync : false;
}

u32 VulkanContext::get_render_width() const
{
    return (m_impl != nullptr) ? m_impl->render_width : 0;
}

u32 VulkanContext::get_render_height() const
{
    return (m_impl != nullptr) ? m_impl->render_height : 0;
}

f32 VulkanContext::get_aspect_ratio() const
{
    if (m_impl == nullptr || m_impl->render_height == 0) return 16.0f / 9.0f;
    return static_cast<f32>(m_impl->render_width) / static_cast<f32>(m_impl->render_height);
}

u32 VulkanContext::get_current_frame_index() const
{
    return (m_impl != nullptr) ? m_impl->current_frame_index : 0;
}

GpuMemoryStats VulkanContext::get_memory_stats() const
{
    GpuMemoryStats stats {};
    if (m_impl == nullptr || m_impl->allocator == VK_NULL_HANDLE) return stats;

    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets {};
    vmaGetHeapBudgets(m_impl->allocator, budgets.data());

    VkPhysicalDeviceMemoryProperties mem_props {};
    vkGetPhysicalDeviceMemoryProperties(m_impl->vkb_physical_device.physical_device, &mem_props);

    for (u32 i = 0; i < mem_props.memoryHeapCount; ++i)
    {
        if ((mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
        {
            stats.allocated_bytes += budgets[i].statistics.blockBytes;
            stats.used_bytes += budgets[i].statistics.allocationBytes;
        }
    }

    stats.active_buffers = m_impl->buffers.active_count();
    stats.active_textures = m_impl->textures.active_count();
    stats.draw_calls_last_frame = m_impl->draw_calls_last_frame;
    stats.triangles_last_frame = m_impl->triangles_last_frame;
    return stats;
}

std::string_view VulkanContext::get_gpu_name() const
{
    if (m_impl == nullptr) return "";
    return m_impl->vkb_physical_device.properties.deviceName;
}

f32 VulkanContext::get_gpu_render_time_ms() const
{
    return (m_impl != nullptr) ? m_impl->gpu_render_time_ms : 0.0f;
}

bool VulkanContext::init_imgui(SDL_Window* window)
{
    if (m_impl == nullptr || !m_impl->initialized || window == nullptr) return false;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr; // Do not create imgui.ini

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.09f, 0.12f, 0.85f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.25f, 0.32f, 0.45f, 0.65f);

    if (!ImGui_ImplSDL3_InitForVulkan(window))
    {
        std::println(stderr, "[VulkanContext] Failed to initialize ImGui SDL3 backend");
        return false;
    }

    ImGui_ImplVulkan_InitInfo init_info {};
    init_info.ApiVersion = VK_API_VERSION_1_3;
    init_info.Instance = m_impl->vkb_instance.instance;
    init_info.PhysicalDevice = m_impl->vkb_physical_device.physical_device;
    init_info.Device = m_impl->device;
    init_info.QueueFamily = m_impl->graphics_queue_family;
    init_info.Queue = m_impl->graphics_queue;
    init_info.DescriptorPoolSize = 64;
    init_info.MinImageCount = 2;
    init_info.ImageCount = static_cast<u32>(m_impl->frames.size());
    init_info.UseDynamicRendering = true;

    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_impl->swapchain_format;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = m_impl->depth_format;

    if (!ImGui_ImplVulkan_Init(&init_info))
    {
        std::println(stderr, "[VulkanContext] Failed to initialize ImGui Vulkan backend");
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    return true;
}

void VulkanContext::shutdown_imgui()
{
    if (ImGui::GetCurrentContext() != nullptr)
    {
        wait_idle();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }
}

void VulkanContext::begin_imgui_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void VulkanContext::end_imgui_frame()
{
    ImGui::Render();
}

void VulkanContext::render_imgui()
{
    if (m_impl == nullptr || !m_impl->rendering_active) return;
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data != nullptr) ImGui_ImplVulkan_RenderDrawData(draw_data, m_impl->frames[m_impl->current_frame_index].cmd);
}

bool VulkanContext::begin_frame()
{
    if (m_impl == nullptr || !m_impl->initialized || m_impl->headless) return false;

    if (m_impl->window != nullptr)
    {
        i32 w = 0;
        i32 h = 0;
        SDL_GetWindowSizeInPixels(m_impl->window, &w, &h);
        if (w <= 0 || h <= 0) return false; // Window is minimized
    }

    if (m_impl->swapchain_needs_rebuild)
    {
        if (!m_impl->build_swapchain(0, 0)) return false;
    }

    FrameData& frame = m_impl->frames[m_impl->current_frame_index];

    if (vkWaitForFences(m_impl->device, 1, &frame.in_flight_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;

    if (frame.timestamp_query_pool != VK_NULL_HANDLE && frame.timestamp_query_active)
    {
        u64 timestamps[2] = {0, 0};
        const VkResult res = vkGetQueryPoolResults(
            m_impl->device, frame.timestamp_query_pool, 0, 2, sizeof(timestamps), timestamps, sizeof(u64), VK_QUERY_RESULT_64_BIT
        );
        if (res == VK_SUCCESS && timestamps[1] >= timestamps[0])
        {
            const f64 duration_ns = static_cast<f64>(timestamps[1] - timestamps[0]) * static_cast<f64>(m_impl->timestamp_period);
            const f32 ms = static_cast<f32>(duration_ns * 1e-6);
            if (m_impl->gpu_render_time_ms <= 0.001f)
                m_impl->gpu_render_time_ms = ms;
            else
                m_impl->gpu_render_time_ms = m_impl->gpu_render_time_ms * 0.85f + ms * 0.15f;
        }
        frame.timestamp_query_active = false;
    }

    const VkResult acquire_res = vkAcquireNextImageKHR(
        m_impl->device, m_impl->vkb_swapchain.swapchain, UINT64_MAX, frame.image_available_semaphore, VK_NULL_HANDLE,
        &m_impl->current_swapchain_image
    );

    if (acquire_res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        m_impl->swapchain_needs_rebuild = true;
        return false;
    }
    if (acquire_res != VK_SUCCESS && acquire_res != VK_SUBOPTIMAL_KHR) return false;

    if (vkResetFences(m_impl->device, 1, &frame.in_flight_fence) != VK_SUCCESS) return false;

    m_impl->flush_frame_deletions(frame);
    frame.staging_offset = 0;

    if (vkResetCommandPool(m_impl->device, frame.cmd_pool, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(frame.cmd, &begin_info) != VK_SUCCESS) return false;

    m_impl->draw_calls_current_frame = 0;
    m_impl->triangles_current_frame = 0;
    m_impl->frame_active = true;
    return true;
}

void VulkanContext::begin_rendering(const RenderPassInfo& pass_info)
{
    if (m_impl == nullptr || !m_impl->frame_active || m_impl->rendering_active) return;

    VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
    VkImage swap_image = m_impl->swapchain_images[m_impl->current_swapchain_image];
    VkImageView swap_view = m_impl->swapchain_image_views[m_impl->current_swapchain_image];

    std::array<VkImageMemoryBarrier2, 2> barriers {};

    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].srcAccessMask = 0;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].image = swap_image;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barriers[1].image = m_impl->depth_attachment.image;
    barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep_info {};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = static_cast<u32>(barriers.size());
    dep_info.pImageMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep_info);

    VkRenderingAttachmentInfo color_attachment {};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = swap_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = pass_info.should_clear_color ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue.color = {
        {pass_info.clear_color[0], pass_info.clear_color[1], pass_info.clear_color[2], pass_info.clear_color[3]}
    };

    VkRenderingAttachmentInfo depth_attachment {};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = m_impl->depth_attachment.view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = pass_info.should_clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue.depthStencil = {pass_info.clear_depth, 0};

    VkRenderingInfo render_info {};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea = {{0, 0}, {m_impl->render_width, m_impl->render_height}};
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments = &color_attachment;
    render_info.pDepthAttachment = &depth_attachment;

    FrameData& frame = m_impl->frames[m_impl->current_frame_index];
    if (frame.timestamp_query_pool != VK_NULL_HANDLE) vkCmdResetQueryPool(cmd, frame.timestamp_query_pool, 0, 2);

    vkCmdBeginRendering(cmd, &render_info);
    m_impl->rendering_active = true;

    if (frame.timestamp_query_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame.timestamp_query_pool, 0);
        frame.timestamp_query_active = true;
    }

    set_viewport(0.0f, 0.0f, static_cast<f32>(m_impl->render_width), static_cast<f32>(m_impl->render_height), 0.0f, 1.0f);
    set_scissor(0, 0, m_impl->render_width, m_impl->render_height);
}

void VulkanContext::end_rendering()
{
    if (m_impl == nullptr || !m_impl->rendering_active) return;

    FrameData& frame = m_impl->frames[m_impl->current_frame_index];
    VkCommandBuffer cmd = frame.cmd;
    if (frame.timestamp_query_pool != VK_NULL_HANDLE && frame.timestamp_query_active)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame.timestamp_query_pool, 1);

    vkCmdEndRendering(cmd);
    m_impl->rendering_active = false;

    VkImage swap_image = m_impl->swapchain_images[m_impl->current_swapchain_image];

    VkImageMemoryBarrier2 present_barrier {};
    present_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    present_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    present_barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    present_barrier.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    present_barrier.dstAccessMask = 0;
    present_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    present_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    present_barrier.image = swap_image;
    present_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep_info {};
    dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep_info.imageMemoryBarrierCount = 1;
    dep_info.pImageMemoryBarriers = &present_barrier;
    vkCmdPipelineBarrier2(cmd, &dep_info);
}

void VulkanContext::end_frame()
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    if (m_impl->rendering_active) end_rendering();

    FrameData& frame = m_impl->frames[m_impl->current_frame_index];
    VkSemaphore render_finished_sem = m_impl->render_finished_semaphores[m_impl->current_swapchain_image];

    vkEndCommandBuffer(frame.cmd);

    VkSemaphoreSubmitInfo wait_sem {};
    wait_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_sem.semaphore = frame.image_available_semaphore;
    wait_sem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signal_sem {};
    signal_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_sem.semaphore = render_finished_sem;
    signal_sem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;

    VkCommandBufferSubmitInfo cmd_submit {};
    cmd_submit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    cmd_submit.commandBuffer = frame.cmd;

    VkSubmitInfo2 submit_info {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit_info.waitSemaphoreInfoCount = 1;
    submit_info.pWaitSemaphoreInfos = &wait_sem;
    submit_info.commandBufferInfoCount = 1;
    submit_info.pCommandBufferInfos = &cmd_submit;
    submit_info.signalSemaphoreInfoCount = 1;
    submit_info.pSignalSemaphoreInfos = &signal_sem;

    vkQueueSubmit2(m_impl->graphics_queue, 1, &submit_info, frame.in_flight_fence);

    VkPresentInfoKHR present_info {};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_sem;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &m_impl->vkb_swapchain.swapchain;
    present_info.pImageIndices = &m_impl->current_swapchain_image;

    const VkResult present_res = vkQueuePresentKHR(m_impl->graphics_queue, &present_info);
    if (present_res == VK_ERROR_OUT_OF_DATE_KHR || present_res == VK_SUBOPTIMAL_KHR) m_impl->swapchain_needs_rebuild = true;

    m_impl->draw_calls_last_frame = m_impl->draw_calls_current_frame;
    m_impl->triangles_last_frame = m_impl->triangles_current_frame;
    m_impl->frame_active = false;
    m_impl->current_frame_index = (m_impl->current_frame_index + 1) % static_cast<u32>(m_impl->frames.size());
}

BufferHandle VulkanContext::create_buffer(u64 size_bytes, BufferUsage usage, MemoryUsage memory_usage, const void* initial_data)
{
    if (m_impl == nullptr || size_bytes == 0) return BufferHandle::invalid();

    GpuBuffer gpu_buf {};
    if (!m_impl->allocate_raw_buffer(size_bytes, usage, memory_usage, gpu_buf)) return BufferHandle::invalid();

    const BufferHandle handle = m_impl->buffers.insert(std::move(gpu_buf));

    if (initial_data != nullptr) upload_buffer_data(handle, 0, initial_data, static_cast<usize>(size_bytes));

    return handle;
}

void VulkanContext::destroy_buffer(BufferHandle handle)
{
    if (m_impl == nullptr) return;

    GpuBuffer removed {};
    if (!m_impl->buffers.extract(handle, removed)) return;

    if (m_impl->frame_active)
        m_impl->frames[m_impl->current_frame_index].pending_buffer_deletions.push_back(removed);
    else
        m_impl->destroy_raw_buffer(removed);
}

bool VulkanContext::upload_buffer_data(BufferHandle handle, u64 offset_bytes, const void* data, usize size_bytes)
{
    if (data == nullptr || size_bytes == 0) return false;
    const auto* byte_ptr = static_cast<const u8*>(data);
    return upload_buffer_data(handle, offset_bytes, std::span<const u8>(byte_ptr, size_bytes));
}

bool VulkanContext::upload_buffer_data(BufferHandle handle, u64 offset_bytes, std::span<const u8> data)
{
    if (m_impl == nullptr || data.empty()) return false;

    GpuBuffer* dst_buf = m_impl->buffers.get(handle);
    if (dst_buf == nullptr || offset_bytes + data.size() > dst_buf->size) return false;

    // Fast path for persistently mapped buffers
    if (dst_buf->mapped_ptr != nullptr)
    {
        std::memcpy(static_cast<u8*>(dst_buf->mapped_ptr) + offset_bytes, data.data(), data.size());
        vmaFlushAllocation(m_impl->allocator, dst_buf->allocation, offset_bytes, data.size());
        return true;
    }

    if (m_impl->frame_active && !m_impl->rendering_active)
    {
        FrameData& frame = m_impl->frames[m_impl->current_frame_index];
        const u64 aligned_offset = (frame.staging_offset + 15ull) & ~15ull;

        if (frame.staging_buffer.mapped_ptr != nullptr && aligned_offset + data.size() <= frame.staging_buffer.size)
        {
            std::memcpy(static_cast<u8*>(frame.staging_buffer.mapped_ptr) + aligned_offset, data.data(), data.size());
            vmaFlushAllocation(m_impl->allocator, frame.staging_buffer.allocation, aligned_offset, data.size());

            VkBufferCopy copy_region {};
            copy_region.srcOffset = aligned_offset;
            copy_region.dstOffset = offset_bytes;
            copy_region.size = data.size();
            vkCmdCopyBuffer(frame.cmd, frame.staging_buffer.buffer, dst_buf->buffer, 1, &copy_region);

            frame.staging_offset = aligned_offset + data.size();
            return true;
        }
    }

    // Synchronous fallback upload for large chunks or initialization-time uploads
    GpuBuffer temp_staging {};
    if (!m_impl->allocate_raw_buffer(data.size(), BufferUsage::TransferSrc, MemoryUsage::CpuToGpu, temp_staging)) return false;

    std::memcpy(temp_staging.mapped_ptr, data.data(), data.size());
    vmaFlushAllocation(m_impl->allocator, temp_staging.allocation, 0, data.size());

    const VkBuffer dst_vk_buf = dst_buf->buffer;
    const u64 copy_size = data.size();

    const bool ok = m_impl->immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy copy_region {};
        copy_region.srcOffset = 0;
        copy_region.dstOffset = offset_bytes;
        copy_region.size = copy_size;
        vkCmdCopyBuffer(cmd, temp_staging.buffer, dst_vk_buf, 1, &copy_region);
    });

    m_impl->destroy_raw_buffer(temp_staging);
    return ok;
}

void* VulkanContext::map_buffer(BufferHandle handle)
{
    if (m_impl == nullptr) return nullptr;
    GpuBuffer* buf = m_impl->buffers.get(handle);
    if (buf == nullptr) return nullptr;
    if (buf->mapped_ptr != nullptr) return buf->mapped_ptr;

    void* mapped = nullptr;
    if (vmaMapMemory(m_impl->allocator, buf->allocation, &mapped) == VK_SUCCESS) return mapped;
    return nullptr;
}

void VulkanContext::unmap_buffer(BufferHandle handle)
{
    if (m_impl == nullptr) return;
    GpuBuffer* buf = m_impl->buffers.get(handle);
    if (buf == nullptr || buf->mapped_ptr != nullptr) return; // Persistently mapped buffers stay mapped
    vmaUnmapMemory(m_impl->allocator, buf->allocation);
}

u64 VulkanContext::get_buffer_device_address(BufferHandle handle) const
{
    if (m_impl == nullptr) return 0;
    const GpuBuffer* buf = m_impl->buffers.get(handle);
    return (buf != nullptr) ? buf->device_address : 0;
}

u64 VulkanContext::get_buffer_size(BufferHandle handle) const
{
    if (m_impl == nullptr) return 0;
    const GpuBuffer* buf = m_impl->buffers.get(handle);
    return (buf != nullptr) ? buf->size : 0;
}

void VulkanContext::copy_buffer(BufferHandle src, BufferHandle dst, u64 size_bytes, u64 src_offset, u64 dst_offset)
{
    if (m_impl == nullptr || size_bytes == 0) return;

    const GpuBuffer* src_buf = m_impl->buffers.get(src);
    const GpuBuffer* dst_buf = m_impl->buffers.get(dst);
    if (src_buf == nullptr || dst_buf == nullptr) return;

    VkBufferCopy region {};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size_bytes;

    if (m_impl->frame_active && !m_impl->rendering_active)
    {
        VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
        vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &region);
    }
    else
    {
        m_impl->immediate_submit([&](VkCommandBuffer cmd) {
            vkCmdCopyBuffer(cmd, src_buf->buffer, dst_buf->buffer, 1, &region);
        });
    }
}

void VulkanContext::fill_buffer(BufferHandle dst, u32 value, u64 size_bytes, u64 offset_bytes)
{
    if (m_impl == nullptr || size_bytes == 0) return;

    const GpuBuffer* dst_buf = m_impl->buffers.get(dst);
    if (dst_buf == nullptr) return;

    if (m_impl->frame_active && !m_impl->rendering_active)
    {
        VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
        vkCmdFillBuffer(cmd, dst_buf->buffer, offset_bytes, size_bytes, value);
    }
    else
    {
        m_impl->immediate_submit([&](VkCommandBuffer cmd) {
            vkCmdFillBuffer(cmd, dst_buf->buffer, offset_bytes, size_bytes, value);
        });
    }
}

BufferHandle VulkanContext::get_or_create_quad_index_buffer(u32 max_quads)
{
    if (m_impl == nullptr || max_quads == 0) return BufferHandle::invalid();

    if (m_impl->quad_index_buffer.is_valid())
    {
        if (m_impl->quad_index_capacity >= max_quads) return m_impl->quad_index_buffer;
        destroy_buffer(m_impl->quad_index_buffer);
    }

    std::vector<u32> indices(static_cast<usize>(max_quads) * 6u);
    for (u32 q = 0; q < max_quads; ++q)
    {
        const u32 base_v = q * 4u;
        const usize base_i = static_cast<usize>(q) * 6u;
        indices[base_i + 0] = base_v + 0u;
        indices[base_i + 1] = base_v + 1u;
        indices[base_i + 2] = base_v + 2u;
        indices[base_i + 3] = base_v + 2u;
        indices[base_i + 4] = base_v + 3u;
        indices[base_i + 5] = base_v + 0u;
    }

    m_impl->quad_index_buffer = create_buffer(
        indices.size() * sizeof(u32), BufferUsage::Index | BufferUsage::TransferDst, MemoryUsage::GpuOnly, indices.data()
    );
    m_impl->quad_index_capacity = max_quads;
    return m_impl->quad_index_buffer;
}


ShaderHandle VulkanContext::create_shader_from_file(std::string_view filepath)
{
    if (m_impl == nullptr || filepath.empty()) return ShaderHandle::invalid();

    std::ifstream file(std::string(filepath), std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        // Try relative to the executable directory
        if (const char* base_path = SDL_GetBasePath())
        {
            const std::string full_path = std::string(base_path) + std::string(filepath);
            file.open(full_path, std::ios::ate | std::ios::binary);
        }
    }

    if (!file.is_open())
    {
        std::println(stderr, "[VulkanContext] Failed to open SPIR-V shader file: {}", filepath);
        return ShaderHandle::invalid();
    }

    const auto file_size = static_cast<usize>(file.tellg());
    if (file_size == 0 || (file_size % sizeof(u32)) != 0)
    {
        std::println(stderr, "[VulkanContext] Invalid SPIR-V file size for: {}", filepath);
        return ShaderHandle::invalid();
    }

    std::vector<u32> words(file_size / sizeof(u32));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(file_size));
    file.close();

    return create_shader_from_spirv(words);
}

ShaderHandle VulkanContext::create_shader_from_spirv(std::span<const u32> spirv_words)
{
    if (m_impl == nullptr || spirv_words.empty()) return ShaderHandle::invalid();

    VkShaderModuleCreateInfo create_info {};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = spirv_words.size_bytes();
    create_info.pCode = spirv_words.data();

    GpuShader shader {};
    if (vkCreateShaderModule(m_impl->device, &create_info, nullptr, &shader.module) != VK_SUCCESS)
    {
        std::println(stderr, "[VulkanContext] Failed to create VkShaderModule");
        return ShaderHandle::invalid();
    }

    parse_spirv_entry_points(spirv_words, shader);
    return m_impl->shaders.insert(std::move(shader));
}

void VulkanContext::destroy_shader(ShaderHandle handle)
{
    if (m_impl == nullptr) return;

    GpuShader removed {};
    if (m_impl->shaders.extract(handle, removed) && removed.module != VK_NULL_HANDLE)
        vkDestroyShaderModule(m_impl->device, removed.module, nullptr);
}

PipelineHandle VulkanContext::create_graphics_pipeline(const GraphicsPipelineDesc& desc)
{
    if (m_impl == nullptr) return PipelineHandle::invalid();

    const GpuShader* vert_shader = m_impl->shaders.get(desc.vertex_shader);
    const ShaderHandle frag_handle = desc.fragment_shader.is_valid() ? desc.fragment_shader : desc.vertex_shader;
    const GpuShader* frag_shader = m_impl->shaders.get(frag_handle);

    if (vert_shader == nullptr || frag_shader == nullptr)
    {
        std::println(stderr, "[VulkanContext] Invalid shader handles in GraphicsPipelineDesc");
        return PipelineHandle::invalid();
    }

    const std::string vert_entry =
        !vert_shader->vertex_entry.empty() ? vert_shader->vertex_entry : std::string(desc.vertex_entry);
    const std::string frag_entry =
        !frag_shader->fragment_entry.empty() ? frag_shader->fragment_entry : std::string(desc.fragment_entry);

    std::array<VkPipelineShaderStageCreateInfo, 2> stages {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_shader->module;
    stages[0].pName = vert_entry.c_str();

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag_shader->module;
    stages[1].pName = frag_entry.c_str();

    VkVertexInputBindingDescription binding_desc {};
    std::vector<VkVertexInputAttributeDescription> attr_descs;

    VkPipelineVertexInputStateCreateInfo vertex_input {};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    if (!desc.vertex_attributes.empty() && desc.vertex_stride > 0)
    {
        binding_desc.binding = 0;
        binding_desc.stride = desc.vertex_stride;
        binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attr_descs.reserve(desc.vertex_attributes.size());
        for (const VertexAttributeDesc& attr : desc.vertex_attributes)
        {
            attr_descs.push_back(
                VkVertexInputAttributeDescription {
                    .location = attr.location, .binding = 0, .format = to_vk_vertex_format(attr.format), .offset = attr.offset
                }
            );
        }

        vertex_input.vertexBindingDescriptionCount = 1;
        vertex_input.pVertexBindingDescriptions = &binding_desc;
        vertex_input.vertexAttributeDescriptionCount = static_cast<u32>(attr_descs.size());
        vertex_input.pVertexAttributeDescriptions = attr_descs.data();
    }

    VkPipelineInputAssemblyStateCreateInfo input_assembly {};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = to_vk_topology(desc.topology);
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state {};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = to_vk_polygon_mode(desc.polygon_mode);
    rasterizer.cullMode = to_vk_cull_mode(desc.cull_mode);
    rasterizer.frontFace = to_vk_front_face(desc.front_face);
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil {};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = desc.depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = desc.depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = to_vk_compare_op(desc.depth_compare);
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend_attachment {};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_attachment.blendEnable = desc.blend_enable ? VK_TRUE : VK_FALSE;
    if (desc.blend_enable)
    {
        blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo color_blending {};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &blend_attachment;

    constexpr std::array<VkDynamicState, 2> dynamic_states {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_info {};
    dynamic_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_info.dynamicStateCount = static_cast<u32>(dynamic_states.size());
    dynamic_info.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering_info {};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &m_impl->swapchain_format;
    rendering_info.depthAttachmentFormat = m_impl->depth_format;

    VkGraphicsPipelineCreateInfo pipeline_info {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &rendering_info;
    pipeline_info.stageCount = static_cast<u32>(stages.size());
    pipeline_info.pStages = stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_info;
    pipeline_info.layout = m_impl->global_pipeline_layout;

    GpuPipeline gpu_pipe {};
    gpu_pipe.layout = m_impl->global_pipeline_layout;
    gpu_pipe.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

    if (vkCreateGraphicsPipelines(m_impl->device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &gpu_pipe.pipeline) != VK_SUCCESS)
    {
        std::println(stderr, "[VulkanContext] Failed to create graphics pipeline");
        return PipelineHandle::invalid();
    }

    return m_impl->pipelines.insert(std::move(gpu_pipe));
}


void VulkanContext::destroy_pipeline(PipelineHandle handle)
{
    if (m_impl == nullptr) return;

    GpuPipeline removed {};
    if (m_impl->pipelines.extract(handle, removed) && removed.pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_impl->device, removed.pipeline, nullptr);
}

void VulkanContext::bind_pipeline(PipelineHandle pipeline)
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    const GpuPipeline* gpu_pipe = m_impl->pipelines.get(pipeline);
    if (gpu_pipe == nullptr) return;

    VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
    m_impl->active_bind_point = gpu_pipe->bind_point;

    vkCmdBindPipeline(cmd, gpu_pipe->bind_point, gpu_pipe->pipeline);
    vkCmdBindDescriptorSets(cmd, gpu_pipe->bind_point, gpu_pipe->layout, 0, 1, &m_impl->bindless_set, 0, nullptr);
}

void VulkanContext::push_constants(const void* data, u32 size_bytes, u32 offset_bytes)
{
    if (m_impl == nullptr || !m_impl->frame_active || data == nullptr || size_bytes == 0) return;

    const u32 clamped_size = std::min(size_bytes, 128u - offset_bytes);
    VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
    vkCmdPushConstants(cmd, m_impl->global_pipeline_layout, VK_SHADER_STAGE_ALL, offset_bytes, clamped_size, data);
}


void VulkanContext::bind_index_buffer_u32(BufferHandle buffer, u64 offset_bytes)
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    const GpuBuffer* gpu_buf = m_impl->buffers.get(buffer);
    if (gpu_buf == nullptr) return;

    VkCommandBuffer cmd = m_impl->frames[m_impl->current_frame_index].cmd;
    vkCmdBindIndexBuffer(cmd, gpu_buf->buffer, offset_bytes, VK_INDEX_TYPE_UINT32);
}

void VulkanContext::set_scissor(i32 x, i32 y, u32 width, u32 height)
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    VkRect2D scissor {.offset = {x, y}, .extent = {width, height}};
    vkCmdSetScissor(m_impl->frames[m_impl->current_frame_index].cmd, 0, 1, &scissor);
}

void VulkanContext::set_viewport(f32 x, f32 y, f32 width, f32 height, f32 min_depth, f32 max_depth)
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    VkViewport viewport {.x = x, .y = y, .width = width, .height = height, .minDepth = min_depth, .maxDepth = max_depth};
    vkCmdSetViewport(m_impl->frames[m_impl->current_frame_index].cmd, 0, 1, &viewport);
}

void VulkanContext::draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance)
{
    if (m_impl == nullptr || !m_impl->rendering_active) return;

    vkCmdDraw(m_impl->frames[m_impl->current_frame_index].cmd, vertex_count, instance_count, first_vertex, first_instance);
    ++m_impl->draw_calls_current_frame;
    m_impl->triangles_current_frame += (vertex_count / 3u) * instance_count;
}

void VulkanContext::draw_indexed(u32 index_count, u32 instance_count, u32 first_index, i32 vertex_offset, u32 first_instance)
{
    if (m_impl == nullptr || !m_impl->rendering_active) return;

    vkCmdDrawIndexed(
        m_impl->frames[m_impl->current_frame_index].cmd, index_count, instance_count, first_index, vertex_offset, first_instance
    );
    ++m_impl->draw_calls_current_frame;
    m_impl->triangles_current_frame += (index_count / 3u) * instance_count;
}

void VulkanContext::draw_indirect(BufferHandle indirect_buffer, u64 offset_bytes, u32 draw_count, u32 stride)
{
    if (m_impl == nullptr || !m_impl->rendering_active || draw_count == 0) return;

    const GpuBuffer* ind_buf = m_impl->buffers.get(indirect_buffer);
    if (ind_buf == nullptr) return;

    vkCmdDrawIndirect(m_impl->frames[m_impl->current_frame_index].cmd, ind_buf->buffer, offset_bytes, draw_count, stride);
    ++m_impl->draw_calls_current_frame;
}

void VulkanContext::draw_indexed_indirect(BufferHandle indirect_buffer, u64 offset_bytes, u32 draw_count, u32 stride)
{
    if (m_impl == nullptr || !m_impl->rendering_active || draw_count == 0) return;

    const GpuBuffer* ind_buf = m_impl->buffers.get(indirect_buffer);
    if (ind_buf == nullptr) return;

    vkCmdDrawIndexedIndirect(m_impl->frames[m_impl->current_frame_index].cmd, ind_buf->buffer, offset_bytes, draw_count, stride);
    ++m_impl->draw_calls_current_frame;
}

void VulkanContext::draw_indirect_count(
    BufferHandle indirect_buffer, u64 offset_bytes, BufferHandle count_buffer, u64 count_offset_bytes, u32 max_draw_count,
    u32 stride
)
{
    if (m_impl == nullptr || !m_impl->rendering_active || max_draw_count == 0) return;

    const GpuBuffer* ind_buf = m_impl->buffers.get(indirect_buffer);
    const GpuBuffer* cnt_buf = m_impl->buffers.get(count_buffer);
    if (ind_buf == nullptr || cnt_buf == nullptr) return;

    vkCmdDrawIndirectCount(
        m_impl->frames[m_impl->current_frame_index].cmd, ind_buf->buffer, offset_bytes, cnt_buf->buffer, count_offset_bytes,
        max_draw_count, stride
    );
    ++m_impl->draw_calls_current_frame;
}

void VulkanContext::draw_indexed_indirect_count(
    BufferHandle indirect_buffer, u64 offset_bytes, BufferHandle count_buffer, u64 count_offset_bytes, u32 max_draw_count,
    u32 stride
)
{
    if (m_impl == nullptr || !m_impl->rendering_active || max_draw_count == 0) return;

    const GpuBuffer* ind_buf = m_impl->buffers.get(indirect_buffer);
    const GpuBuffer* cnt_buf = m_impl->buffers.get(count_buffer);
    if (ind_buf == nullptr || cnt_buf == nullptr) return;

    vkCmdDrawIndexedIndirectCount(
        m_impl->frames[m_impl->current_frame_index].cmd, ind_buf->buffer, offset_bytes, cnt_buf->buffer, count_offset_bytes,
        max_draw_count, stride
    );
    ++m_impl->draw_calls_current_frame;
}

void VulkanContext::dispatch_compute(u32 group_count_x, u32 group_count_y, u32 group_count_z)
{
    if (m_impl == nullptr || !m_impl->frame_active || m_impl->rendering_active) return;

    vkCmdDispatch(m_impl->frames[m_impl->current_frame_index].cmd, group_count_x, group_count_y, group_count_z);
}

void VulkanContext::dispatch_compute_indirect(BufferHandle indirect_buffer, u64 offset_bytes)
{
    if (m_impl == nullptr || !m_impl->frame_active || m_impl->rendering_active) return;

    const GpuBuffer* ind_buf = m_impl->buffers.get(indirect_buffer);
    if (ind_buf == nullptr) return;

    vkCmdDispatchIndirect(m_impl->frames[m_impl->current_frame_index].cmd, ind_buf->buffer, offset_bytes);
}

void VulkanContext::memory_barrier_transfer_to_graphics()
{
    if (m_impl == nullptr || !m_impl->frame_active) return;

    VkMemoryBarrier2 barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

    VkDependencyInfo dep {};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(m_impl->frames[m_impl->current_frame_index].cmd, &dep);
}

} // namespace vulk