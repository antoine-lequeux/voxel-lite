#include "engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>
#include <glm/glm.hpp>

#include <array>
#include <format>
#include <print>

namespace vl
{

namespace
{

// Matches CubeVertex in cube.slang
struct CubeVertex
{
    glm::vec3 position;
    f32 ao;
    glm::vec3 normal;
    u32 face_id;
    glm::vec3 color;
    f32 roughness;
    glm::vec2 uv;
    glm::vec2 pad;
};

static_assert(sizeof(CubeVertex) == 64, "CubeVertex must match 64-byte shader layout");

// Matches PushConstants in cube.slang
struct CubePushConstants
{
    glm::mat4 mvp;
    glm::vec4 model_offset_and_time;
    glm::vec4 camera_pos;
    u64 vertex_buffer_address;
    u64 padding;
};

static_assert(sizeof(CubePushConstants) <= 128, "CubePushConstants must fit within 128 bytes");

} // namespace

bool Engine::init()
{
    std::println("[Engine] Initializing SDL3 and VulkanContext...");

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::println(stderr, "[Engine] SDL_Init failed: {}", SDL_GetError());
        return false;
    }

    std::string window_title = std::format(
        "{} v{}.{}.f{}", AppConfig::NAME, AppConfig::VERSION_MAJOR, AppConfig::VERSION_MINOR, AppConfig::VERSION_PATCH
    );
    m_window = SDL_CreateWindow(window_title.c_str(), 1280, 720, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    if (m_window == nullptr)
    {
        std::println(stderr, "[Engine] SDL_CreateWindow failed: {}", SDL_GetError());
        return false;
    }

    vulk::ContextConfig ctx_config {};
    ctx_config.app_name = AppConfig::NAME;
    ctx_config.app_version_major = AppConfig::VERSION_MAJOR;
    ctx_config.app_version_minor = AppConfig::VERSION_MINOR;
    ctx_config.app_version_patch = AppConfig::VERSION_PATCH;
    ctx_config.enable_validation = true;
    ctx_config.vsync = false;
    ctx_config.frames_in_flight = 2;

    if (!m_vulkan_context.init(m_window, ctx_config))
    {
        std::println(stderr, "[Engine] Failed to initialize VulkanContext");
        return false;
    }

    if (!init_cube_resources())
    {
        std::println(stderr, "[Engine] Failed to initialize cube GPU resources");
        return false;
    }

    m_running = true;
    return true;
}

bool Engine::init_cube_resources()
{
    constexpr glm::vec3 color_grass {0.36f, 0.76f, 0.28f};
    constexpr glm::vec3 color_dirt {0.52f, 0.37f, 0.24f};

    const auto make_face = [](glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 normal, glm::vec3 color,
                              u32 face_id) constexpr -> std::array<CubeVertex, 4> {
        return {{
            {.position = v0,
             .ao = 1.00f,
             .normal = normal,
             .face_id = face_id,
             .color = color,
             .roughness = 0.8f,
             .uv = {0.0f, 0.0f},
             .pad = {0.0f, 0.0f}},
            {.position = v1,
             .ao = 1.00f,
             .normal = normal,
             .face_id = face_id,
             .color = color,
             .roughness = 0.8f,
             .uv = {1.0f, 0.0f},
             .pad = {0.0f, 0.0f}},
            {.position = v2,
             .ao = 1.00f,
             .normal = normal,
             .face_id = face_id,
             .color = color,
             .roughness = 0.8f,
             .uv = {1.0f, 1.0f},
             .pad = {0.0f, 0.0f}},
            {.position = v3,
             .ao = 1.00f,
             .normal = normal,
             .face_id = face_id,
             .color = color,
             .roughness = 0.8f,
             .uv = {0.0f, 1.0f},
             .pad = {0.0f, 0.0f}},
        }};
    };

    // Initialize 6 faces with 4 vertices each
    const std::array<std::array<CubeVertex, 4>, 6> faces {{
        // +Z (front face)
        make_face(
            {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, color_grass, 0
        ),
        // -Z (back face)
        make_face(
            {0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, -1.0f},
            color_grass, 1
        ),
        // +X (right face)
        make_face(
            {0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, color_grass, 2
        ),
        // -X (left face)
        make_face(
            {-0.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f},
            color_grass, 3
        ),
        // +Y (top face)
        make_face(
            {-0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, color_grass, 4
        ),
        // -Y (bottom face)
        make_face(
            {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {0.0f, -1.0f, 0.0f},
            color_dirt, 5
        ),
    }};

    // Upload cube vertices into VRAM with Shader Device Address enabled for vertex pulling
    m_cube_vertex_buffer = m_vulkan_context.create_buffer(
        sizeof(faces), vulk::BufferUsage::Storage | vulk::BufferUsage::ShaderDeviceAddress, vulk::MemoryUsage::GpuOnly,
        faces.data()
    );

    // Use the global shared quad index buffer for cube faces
    m_cube_index_buffer = m_vulkan_context.get_or_create_quad_index_buffer(6);

    if (!m_cube_vertex_buffer.is_valid() || !m_cube_index_buffer.is_valid()) return false;

    m_cube_shader = m_vulkan_context.create_shader_from_file("cube.spirv");
    if (!m_cube_shader.is_valid()) return false;

    vulk::GraphicsPipelineDesc pipe_desc {};
    pipe_desc.vertex_shader = m_cube_shader;
    pipe_desc.fragment_shader = m_cube_shader;
    pipe_desc.vertex_entry = "vertexMain";
    pipe_desc.fragment_entry = "fragmentMain";
    pipe_desc.topology = vulk::PrimitiveTopology::TriangleList;
    pipe_desc.polygon_mode = vulk::PolygonMode::Fill;
    pipe_desc.cull_mode = vulk::CullMode::Back;
    pipe_desc.front_face = vulk::FrontFace::CounterClockwise;
    pipe_desc.depth_test = true;
    pipe_desc.depth_write = true;
    pipe_desc.depth_compare = vulk::CompareOp::LessOrEqual;
    pipe_desc.push_constant_size = sizeof(CubePushConstants);

    m_cube_pipeline = m_vulkan_context.create_graphics_pipeline(pipe_desc);
    return m_cube_pipeline.is_valid();
}

void Engine::process_events()
{
    SDL_Event event {};
    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED: m_running = false; break;

            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                m_vulkan_context.on_resize(static_cast<u32>(event.window.data1), static_cast<u32>(event.window.data2));
                break;

            case SDL_EVENT_KEY_DOWN:
                if (event.key.scancode == SDL_SCANCODE_ESCAPE)
                {
                    if (m_mouse_captured)
                    {
                        m_mouse_captured = false;
                        SDL_SetWindowRelativeMouseMode(m_window, false);
                    }
                    else
                    {
                        m_running = false;
                    }
                }
                break;

            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (event.button.button == SDL_BUTTON_LEFT || event.button.button == SDL_BUTTON_RIGHT)
                {
                    if (!m_mouse_captured)
                    {
                        m_mouse_captured = true;
                        SDL_SetWindowRelativeMouseMode(m_window, true);
                    }
                }
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (m_mouse_captured || (event.motion.state & SDL_BUTTON_RMASK) != 0)
                {
                    m_camera.rotate(
                        event.motion.xrel * m_camera.mouse_sensitivity, -event.motion.yrel * m_camera.mouse_sensitivity
                    );
                }
                break;

            default: break;
        }
    }
}

void Engine::update(f32 dt)
{
    m_elapsed_time += dt;

    i32 num_keys = 0;
    const bool* keys = SDL_GetKeyboardState(&num_keys);
    if (keys == nullptr) return;

    // Player movement
    glm::vec3 move_dir {0.0f, 0.0f, 0.0f};
    const glm::vec3 fwd = m_camera.flat_forward();
    const glm::vec3 rgt = m_camera.right();
    constexpr glm::vec3 world_up {0.0f, 1.0f, 0.0f};

    if (keys[SDL_SCANCODE_W]) move_dir += fwd;
    if (keys[SDL_SCANCODE_S]) move_dir -= fwd;
    if (keys[SDL_SCANCODE_D]) move_dir += rgt;
    if (keys[SDL_SCANCODE_A]) move_dir -= rgt;
    if (keys[SDL_SCANCODE_SPACE]) move_dir += world_up;
    if (keys[SDL_SCANCODE_LSHIFT]) move_dir -= world_up;

    if (glm::dot(move_dir, move_dir) > 0.0001f)
    {
        move_dir = glm::normalize(move_dir);
        const f32 speed = keys[SDL_SCANCODE_LSHIFT] ? (m_camera.move_speed * m_camera.sprint_multiplier) : m_camera.move_speed;
        m_camera.position += move_dir * (speed * dt);
    }
}

void Engine::render()
{
    if (!m_vulkan_context.begin_frame()) return;

    vulk::RenderPassInfo pass_info {};
    pass_info.clear_color = {0.42f, 0.62f, 0.88f, 1.0f}; // Sky color
    pass_info.clear_depth = 1.0f;

    m_vulkan_context.begin_rendering(pass_info);

    m_vulkan_context.bind_pipeline(m_cube_pipeline);
    m_vulkan_context.bind_index_buffer_u32(m_cube_index_buffer);

    CubePushConstants pc {};
    pc.mvp = m_camera.view_projection_matrix(m_vulkan_context.get_aspect_ratio());
    pc.model_offset_and_time = glm::vec4 {0.0f, 0.0f, 0.0f, m_elapsed_time};
    pc.camera_pos = glm::vec4 {m_camera.position, 1.0f};
    pc.vertex_buffer_address = m_vulkan_context.get_buffer_device_address(m_cube_vertex_buffer);
    pc.padding = 0;

    m_vulkan_context.push_constants(pc);
    m_vulkan_context.draw_indexed(36);

    m_vulkan_context.end_rendering();
    m_vulkan_context.end_frame();
}

void Engine::run()
{
    std::println("[Engine] Entering main loop...");

    u64 last_counter = SDL_GetPerformanceCounter();
    const f32 perf_freq = static_cast<f32>(SDL_GetPerformanceFrequency());
    f32 fps_timer = 0.0f;
    u32 frame_count = 0;

    while (m_running)
    {
        const u64 now_counter = SDL_GetPerformanceCounter();
        const f32 dt = std::min(0.1f, static_cast<f32>(now_counter - last_counter) / perf_freq);
        last_counter = now_counter;
        fps_timer += dt;
        frame_count++;

        if (fps_timer >= 1.0f)
        {
            std::println(
                "fps: {:.2f} ({:.2f} ms)", static_cast<f32>(frame_count) / fps_timer,
                (fps_timer * 1000.0f) / static_cast<f32>(frame_count)
            );
            fps_timer = 0.0f;
            frame_count = 0;
        }

        process_events();
        update(dt);
        render();
    }
}

void Engine::shutdown()
{
    std::println("[Engine] Shutting engine down...");

    m_vulkan_context.wait_idle();

    if (m_cube_pipeline.is_valid()) m_vulkan_context.destroy_pipeline(m_cube_pipeline);
    if (m_cube_shader.is_valid()) m_vulkan_context.destroy_shader(m_cube_shader);
    if (m_cube_vertex_buffer.is_valid()) m_vulkan_context.destroy_buffer(m_cube_vertex_buffer);

    m_vulkan_context.shutdown();

    if (m_window != nullptr)
    {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }

    SDL_Quit();
}

} // namespace vl