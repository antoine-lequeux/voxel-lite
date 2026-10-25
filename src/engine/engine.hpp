#pragma once

#include "overlay.hpp"
#include "worldmanager.hpp"

struct SDL_Window;

namespace vl
{

namespace AppConfig
{
inline constexpr const char* NAME = "Voxel Lite";
inline constexpr u32 VERSION_MAJOR = 0;
inline constexpr u32 VERSION_MINOR = 5;
inline constexpr u32 VERSION_PATCH = 0;
} // namespace AppConfig

class Engine
{
public:

    bool init();
    void run();
    void shutdown();

private:

    bool init_chunk_resources();
    bool ensure_draw_buffers_capacity(u32 required_slots);
    void process_events();
    void handle_inputs(f32 dt);
    void render();

    static constexpr u32 FRAMES_IN_FLIGHT = 2;

    SDL_Window* m_window = nullptr;
    vulk::VulkanContext m_vulkan_context {};
    Camera m_camera {};

    std::unique_ptr<WorldManager> m_manager {};

    vulk::BufferHandle m_block_material_buffer {};
    vulk::ShaderHandle m_chunk_shader {};
    vulk::PipelineHandle m_chunk_pipeline {};

    vulk::ShaderHandle m_cull_shader {};
    vulk::PipelineHandle m_cull_pipeline {};
    vulk::BufferHandle m_chunk_draw_buffer {};
    vulk::BufferHandle m_indirect_buffer {};
    std::array<vulk::BufferHandle, FRAMES_IN_FLIGHT> m_cull_stats_buffers {};
    std::array<bool, FRAMES_IN_FLIGHT> m_cull_stats_pending {};
    u32 m_draw_buffers_capacity = 0;
    u32 m_last_chunks_drawn = 0;
    u32 m_last_triangles_drawn = 0;

    f32 m_elapsed_time = 0.0f;
    bool m_running = false;
    bool m_mouse_captured = false;

    DebugOverlay m_overlay {};
};

} // namespace vl