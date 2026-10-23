#include "engine.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_scancode.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>

#include <print>

namespace vl
{

namespace
{

// Matches PushConstants in chunk.slang
struct ChunkPushConstants
{
    glm::mat4 mvp;
    glm::vec4 chunk_origin_and_time;
    glm::vec4 camera_pos;
    u64 vertex_buffer_address;
    u64 material_buffer_address;
};

static_assert(sizeof(ChunkPushConstants) <= 128, "ChunkPushConstants must fit within 128 bytes");

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
#if defined(NDEBUG)
    ctx_config.enable_validation = false;
#else
    ctx_config.enable_validation = true;
#endif
    ctx_config.vsync = false;
    ctx_config.frames_in_flight = 2;
    ctx_config.staging_buffer_size = 48ull * 1024ull * 1024ull;

    if (!m_vulkan_context.init(m_window, ctx_config))
    {
        std::println(stderr, "[Engine] Failed to initialize VulkanContext");
        return false;
    }

    // Position the camera
    m_camera.position = glm::vec3 {16.0f, 25.0f, 44.0f};
    m_camera.yaw_degrees = -90.0f;
    m_camera.pitch_degrees = -24.0f;
    m_camera.move_speed = 12.0f;

    if (!init_chunk_resources())
    {
        std::println(stderr, "[Engine] Failed to initialize chunk GPU resources");
        return false;
    }

    if (!m_vulkan_context.init_imgui(m_window))
    {
        std::println(stderr, "[Engine] Failed to initialize ImGui");
        return false;
    }

    m_manager = std::make_unique<WorldManager>(12345);

    m_running = true;
    return true;
}

bool Engine::init_chunk_resources()
{
    // Upload the block registry material table once to GPU memory
    constexpr auto gpu_materials = build_gpu_block_materials();
    m_block_material_buffer = m_vulkan_context.create_buffer(
        sizeof(gpu_materials), vulk::BufferUsage::Storage | vulk::BufferUsage::ShaderDeviceAddress, vulk::MemoryUsage::GpuOnly,
        gpu_materials.data()
    );
    if (!m_block_material_buffer.is_valid()) return false;

    m_chunk_shader = m_vulkan_context.create_shader_from_file("shaders/chunk.spirv");
    if (!m_chunk_shader.is_valid()) return false;

    vulk::GraphicsPipelineDesc pipe_desc {};
    pipe_desc.vertex_shader = m_chunk_shader;
    pipe_desc.fragment_shader = m_chunk_shader;
    pipe_desc.vertex_entry = "vertexMain";
    pipe_desc.fragment_entry = "fragmentMain";
    pipe_desc.topology = vulk::PrimitiveTopology::TriangleList;
    pipe_desc.polygon_mode = vulk::PolygonMode::Fill;
    pipe_desc.cull_mode = vulk::CullMode::Back;
    pipe_desc.front_face = vulk::FrontFace::CounterClockwise;
    pipe_desc.depth_test = true;
    pipe_desc.depth_write = true;
    pipe_desc.depth_compare = vulk::CompareOp::LessOrEqual;
    pipe_desc.push_constant_size = sizeof(ChunkPushConstants);

    m_chunk_pipeline = m_vulkan_context.create_graphics_pipeline(pipe_desc);
    return m_chunk_pipeline.is_valid();
}

void Engine::process_events()
{
    SDL_Event event {};
    while (SDL_PollEvent(&event))
    {
        ImGui_ImplSDL3_ProcessEvent(&event);

        const ImGuiIO& io = ImGui::GetIO();

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
                    if (!m_mouse_captured && !io.WantCaptureMouse)
                    {
                        m_mouse_captured = true;
                        SDL_SetWindowRelativeMouseMode(m_window, true);
                    }
                }
                break;

            case SDL_EVENT_MOUSE_MOTION:
                if (m_mouse_captured)
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

void Engine::handle_inputs(f32 dt)
{
    const bool* keys = SDL_GetKeyboardState(nullptr);
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
        const f32 speed = keys[SDL_SCANCODE_LCTRL] ? (m_camera.move_speed * m_camera.sprint_multiplier) : m_camera.move_speed;
        m_camera.position += move_dir * (speed * dt);
    }
}

void Engine::render()
{
    vulk::RenderPassInfo pass_info {};
    pass_info.clear_color = {0.47f, 0.65f, 1.0f, 1.0f}; // Sky color
    pass_info.clear_depth = 1.0f;

    m_vulkan_context.begin_rendering(pass_info);

    // Bind graphics pipeline and quad index buffer once per frame
    m_vulkan_context.bind_pipeline(m_chunk_pipeline);

    const vulk::BufferHandle quad_ibo = m_vulkan_context.get_or_create_quad_index_buffer(16384);
    if (quad_ibo.is_valid()) m_vulkan_context.bind_index_buffer_u32(quad_ibo);

    m_camera.update_far_plane(m_manager->get_render_distance(), CHUNK_SIZE);

    const glm::mat4 vp = m_camera.view_projection_matrix(m_vulkan_context.get_aspect_ratio());
    const Frustum frustum = Frustum::from_view_projection(vp);
    const glm::vec4 cam_pos = glm::vec4 {m_camera.position, 1.0f};
    const u64 material_addr = m_vulkan_context.get_buffer_device_address(m_block_material_buffer);

    u32 chunks_drawn = 0;
    u32 chunks_culled = 0;

    // Render active chunks across all LOD pools
    m_manager->for_each_active_mesh([&](const ChunkMesh& mesh) {
        if (mesh.empty() || mesh.device_address == 0u) return;

        const glm::vec3 origin = mesh.world_origin();
        constexpr glm::vec3 chunk_dims {static_cast<f32>(CHUNK_SIZE)};

        // Frustum culling
        if (m_overlay.frustum_culling_enabled && !frustum.intersects_aabb(origin, origin + chunk_dims))
        {
            ++chunks_culled;
            return;
        }

        ChunkPushConstants pc {};
        pc.mvp = vp;
        pc.chunk_origin_and_time = glm::vec4 {origin, m_elapsed_time};
        pc.camera_pos = cam_pos;
        pc.vertex_buffer_address = mesh.device_address;
        pc.material_buffer_address = material_addr;

        m_vulkan_context.push_constants(pc);
        m_vulkan_context.draw_indexed(mesh.index_count());
        ++chunks_drawn;
    });

    m_overlay.set_chunk_culling_stats(chunks_drawn, chunks_culled);

    // Render ImGui overlay on top of 3D geometry
    m_vulkan_context.render_imgui();

    m_vulkan_context.end_rendering();
}

void Engine::run()
{
    std::println("[Engine] Entering main loop...");

    u64 last_counter = SDL_GetPerformanceCounter();
    const f64 perf_freq = static_cast<f64>(SDL_GetPerformanceFrequency());
    f32 last_cpu_work_ms = 0.0f;

    while (m_running)
    {
        const u64 now_counter = SDL_GetPerformanceCounter();
        const f32 dt = std::min(0.1f, static_cast<f32>(static_cast<f64>(now_counter - last_counter) / perf_freq));
        last_counter = now_counter;

        u64 cpu_work_ticks = 0;
        const u64 t0 = SDL_GetPerformanceCounter();

        process_events();
        handle_inputs(dt);
        m_elapsed_time += dt;

        // Update debug overlay and construct ImGui draw lists
        m_overlay.update(dt, last_cpu_work_ms);
        m_vulkan_context.begin_imgui_frame();
        m_overlay.draw(dt, *m_manager, m_camera, m_vulkan_context);
        m_vulkan_context.end_imgui_frame();

        cpu_work_ticks += (SDL_GetPerformanceCounter() - t0);

        if (!m_vulkan_context.begin_frame()) continue;

        const u64 t1 = SDL_GetPerformanceCounter();

        m_manager->update(m_camera.position, m_vulkan_context);
        m_vulkan_context.memory_barrier_transfer_to_graphics();

        render();
        m_vulkan_context.end_frame();

        cpu_work_ticks += (SDL_GetPerformanceCounter() - t1);
        last_cpu_work_ms = static_cast<f32>(static_cast<f64>(cpu_work_ticks) * 1000.0 / perf_freq);
    }
}

void Engine::shutdown()
{
    std::println("[Engine] Shutting engine down...");

    m_vulkan_context.wait_idle();
    m_vulkan_context.shutdown_imgui();

    m_manager->destroy_all_gpu_resources(m_vulkan_context);

    if (m_block_material_buffer.is_valid()) m_vulkan_context.destroy_buffer(m_block_material_buffer);
    if (m_chunk_pipeline.is_valid()) m_vulkan_context.destroy_pipeline(m_chunk_pipeline);
    if (m_chunk_shader.is_valid()) m_vulkan_context.destroy_shader(m_chunk_shader);

    m_vulkan_context.shutdown();

    if (m_window != nullptr)
    {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }

    SDL_Quit();
}

} // namespace vl