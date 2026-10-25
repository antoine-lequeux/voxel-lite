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
    glm::vec4 camera_pos_and_time;
    u64 chunk_draws_address;
    u64 material_buffer_address;
};

static_assert(sizeof(ChunkPushConstants) <= 128, "ChunkPushConstants must fit within 128 bytes");

// Matches CullPushConstants in cull.slang
struct CullPushConstants
{
    std::array<glm::vec4, 6> planes {};
    u64 chunk_draws_address = 0;
    u64 commands_address = 0;
    u64 stats_address = 0;
    u32 slot_count = 0;
    u32 flags = 0;
};

static_assert(sizeof(CullPushConstants) <= 128, "CullPushConstants must fit within 128 bytes");

// Matches CullStats in cull.slang
struct GpuCullStats
{
    u32 draw_count = 0;
    u32 quad_count = 0;
};

constexpr u32 CULL_FLAG_FRUSTUM = 1u;
constexpr u32 CULL_WORKGROUP_SIZE = 64u;
constexpr u32 INITIAL_DRAW_SLOTS_CAPACITY = 65536u;

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
    if (!m_chunk_pipeline.is_valid()) return false;

    m_cull_shader = m_vulkan_context.create_shader_from_file("shaders/cull.spirv");
    if (!m_cull_shader.is_valid()) return false;

    m_cull_pipeline = m_vulkan_context.create_compute_pipeline(m_cull_shader, "cullMain");
    if (!m_cull_pipeline.is_valid()) return false;

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i)
    {
        constexpr GpuCullStats zero_stats {};
        m_cull_stats_buffers[i] = m_vulkan_context.create_buffer(
            sizeof(GpuCullStats),
            vulk::BufferUsage::Storage | vulk::BufferUsage::Indirect | vulk::BufferUsage::TransferDst |
                vulk::BufferUsage::ShaderDeviceAddress,
            vulk::MemoryUsage::GpuToCpu, &zero_stats
        );
        if (!m_cull_stats_buffers[i].is_valid()) return false;
    }

    if (!m_vulkan_context.get_or_create_quad_index_buffer(16384).is_valid()) return false;

    return ensure_draw_buffers_capacity(INITIAL_DRAW_SLOTS_CAPACITY);
}

bool Engine::ensure_draw_buffers_capacity(u32 required_slots)
{
    if (required_slots <= m_draw_buffers_capacity) return true;

    u32 new_capacity = std::max(INITIAL_DRAW_SLOTS_CAPACITY, m_draw_buffers_capacity);
    while (new_capacity < required_slots) new_capacity *= 2u;

    const u64 draw_bytes = static_cast<u64>(new_capacity) * sizeof(GpuChunkDraw);
    const vulk::BufferHandle new_draw_buf = m_vulkan_context.create_buffer(
        draw_bytes,
        vulk::BufferUsage::Storage | vulk::BufferUsage::TransferSrc | vulk::BufferUsage::TransferDst |
            vulk::BufferUsage::ShaderDeviceAddress,
        vulk::MemoryUsage::GpuOnly
    );
    if (!new_draw_buf.is_valid()) return false;

    if (m_chunk_draw_buffer.is_valid())
    {
        const u64 old_bytes = static_cast<u64>(m_draw_buffers_capacity) * sizeof(GpuChunkDraw);
        m_vulkan_context.copy_buffer(m_chunk_draw_buffer, new_draw_buf, old_bytes);
        m_vulkan_context.memory_barrier_graphics_to_transfer();
        m_vulkan_context.destroy_buffer(m_chunk_draw_buffer);
    }
    m_chunk_draw_buffer = new_draw_buf;

    if (m_indirect_buffer.is_valid()) m_vulkan_context.destroy_buffer(m_indirect_buffer);

    const u64 indirect_bytes = static_cast<u64>(new_capacity) * sizeof(vulk::DrawIndexedIndirectCommand);
    m_indirect_buffer = m_vulkan_context.create_buffer(
        indirect_bytes, vulk::BufferUsage::Storage | vulk::BufferUsage::Indirect | vulk::BufferUsage::ShaderDeviceAddress,
        vulk::MemoryUsage::GpuOnly
    );
    if (!m_indirect_buffer.is_valid()) return false;

    m_draw_buffers_capacity = new_capacity;
    return true;
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
    ChunkDrawTable& draw_table = ChunkDrawTable::instance();
    const u32 frame_idx = m_vulkan_context.get_current_frame_index() % FRAMES_IN_FLIGHT;
    const vulk::BufferHandle stats_buf = m_cull_stats_buffers[frame_idx];

    // Read back GPU culling statistics from the previous completed execution of this frame slot
    if (m_cull_stats_pending[frame_idx])
    {
        GpuCullStats stats {};
        if (m_vulkan_context.read_buffer_data(stats_buf, 0, &stats, sizeof(stats)))
        {
            m_last_chunks_drawn = stats.draw_count;
            m_last_triangles_drawn = stats.quad_count * 2u;
        }
        m_cull_stats_pending[frame_idx] = false;
    }

    const u32 active_meshes = draw_table.active_count();
    const u32 slot_limit = draw_table.slot_limit();

    if (slot_limit > 0u) ensure_draw_buffers_capacity(slot_limit);
    if (draw_table.has_dirty_slots()) draw_table.flush_dirty(m_vulkan_context, m_chunk_draw_buffer);

    m_camera.update_far_plane(m_manager->get_render_distance(), CHUNK_SIZE);
    const glm::mat4 vp = m_camera.view_projection_matrix(m_vulkan_context.get_aspect_ratio());
    const Frustum frustum = Frustum::from_view_projection(vp);

    if (active_meshes > 0u && slot_limit > 0u)
    {
        m_vulkan_context.fill_buffer(stats_buf, 0u, sizeof(GpuCullStats), 0u);
        m_vulkan_context.memory_barrier_transfer_to_compute();
        m_vulkan_context.memory_barrier_transfer_to_graphics();

        CullPushConstants cull_pc {};
        cull_pc.planes = frustum.planes;
        cull_pc.chunk_draws_address = m_vulkan_context.get_buffer_device_address(m_chunk_draw_buffer);
        cull_pc.commands_address = m_vulkan_context.get_buffer_device_address(m_indirect_buffer);
        cull_pc.stats_address = m_vulkan_context.get_buffer_device_address(stats_buf);
        cull_pc.slot_count = slot_limit;
        cull_pc.flags = m_overlay.frustum_culling_enabled ? CULL_FLAG_FRUSTUM : 0u;

        m_vulkan_context.bind_pipeline(m_cull_pipeline);
        m_vulkan_context.push_constants(cull_pc);
        m_vulkan_context.dispatch_compute((slot_limit + CULL_WORKGROUP_SIZE - 1u) / CULL_WORKGROUP_SIZE);
        m_vulkan_context.memory_barrier_compute_to_draw();
        m_cull_stats_pending[frame_idx] = true;
    }
    else
    {
        m_last_chunks_drawn = 0;
        m_last_triangles_drawn = 0;
    }

    const u32 total_loaded = m_manager->get_active_chunks_count();
    const u32 chunks_drawn = std::min(m_last_chunks_drawn, active_meshes);
    const u32 chunks_culled = (total_loaded > chunks_drawn) ? (total_loaded - chunks_drawn) : 0u;
    m_overlay.set_chunk_culling_stats(chunks_drawn, chunks_culled);

    vulk::RenderPassInfo pass_info {};
    pass_info.clear_color = {0.47f, 0.65f, 1.0f, 1.0f}; // Sky color
    pass_info.clear_depth = 1.0f;

    m_vulkan_context.begin_rendering(pass_info);

    if (active_meshes > 0u && slot_limit > 0u)
    {
        const vulk::BufferHandle quad_ibo = m_vulkan_context.get_or_create_quad_index_buffer(16384);
        if (quad_ibo.is_valid())
        {
            m_vulkan_context.bind_pipeline(m_chunk_pipeline);
            m_vulkan_context.bind_index_buffer_u32(quad_ibo);

            ChunkPushConstants pc {};
            pc.mvp = vp;
            pc.camera_pos_and_time = glm::vec4 {m_camera.position, m_elapsed_time};
            pc.chunk_draws_address = m_vulkan_context.get_buffer_device_address(m_chunk_draw_buffer);
            pc.material_buffer_address = m_vulkan_context.get_buffer_device_address(m_block_material_buffer);

            m_vulkan_context.push_constants(pc);
            m_vulkan_context.draw_indexed_indirect_count(m_indirect_buffer, 0, stats_buf, 0, active_meshes);
            m_vulkan_context.report_indirect_triangles(m_last_triangles_drawn);
        }
    }

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

        m_vulkan_context.memory_barrier_graphics_to_transfer();
        m_manager->update(m_camera.position, m_vulkan_context);

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
    if (m_chunk_draw_buffer.is_valid()) m_vulkan_context.destroy_buffer(m_chunk_draw_buffer);
    if (m_indirect_buffer.is_valid()) m_vulkan_context.destroy_buffer(m_indirect_buffer);
    for (vulk::BufferHandle& buf : m_cull_stats_buffers)
        if (buf.is_valid()) m_vulkan_context.destroy_buffer(buf);
    if (m_cull_pipeline.is_valid()) m_vulkan_context.destroy_pipeline(m_cull_pipeline);
    if (m_cull_shader.is_valid()) m_vulkan_context.destroy_shader(m_cull_shader);
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