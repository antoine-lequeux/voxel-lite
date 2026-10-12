#pragma once

#include "engine/camera.hpp"
#include "renderer/vulkancontext.hpp"

struct SDL_Window;

namespace vl
{

namespace AppConfig
{
inline constexpr const char* NAME = "Voxel Lite";
inline constexpr u32 VERSION_MAJOR = 0;
inline constexpr u32 VERSION_MINOR = 1;
inline constexpr u32 VERSION_PATCH = 0;
} // namespace AppConfig

class Engine
{
public:

    bool init();
    void run();
    void shutdown();

private:

    bool init_cube_resources();
    void process_events();
    void update(f32 dt);
    void render();

    SDL_Window* m_window = nullptr;
    vulk::VulkanContext m_vulkan_context {};
    Camera m_camera {};

    vulk::BufferHandle m_cube_vertex_buffer {};
    vulk::BufferHandle m_cube_index_buffer {};
    vulk::ShaderHandle m_cube_shader {};
    vulk::PipelineHandle m_cube_pipeline {};

    f32 m_elapsed_time = 0.0f;
    bool m_running = false;
    bool m_mouse_captured = false;
};

} // namespace vl