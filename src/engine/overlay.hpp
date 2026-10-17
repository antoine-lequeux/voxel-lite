#pragma once

#include "engine/camera.hpp"
#include "engine/chunk.hpp"
#include "engine/worldmanager.hpp"
#include "renderer/vulkancontext.hpp"

#include <imgui.h>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
// clang-format off
    #include <windows.h>
    #include <psapi.h>
// clang-format on
#endif

namespace vl
{

struct SystemMetrics
{
    f32 cpu_process_percent = 0.0f;
    f32 ram_process_mb = 0.0f;
    f32 ram_commit_mb = 0.0f;
    f32 ram_sys_used_gb = 0.0f;
    f32 ram_sys_total_gb = 0.0f;
    u32 ram_sys_percent = 0;

    void update(f32 dt)
    {
        m_sample_timer += dt;
        if (m_sample_timer < 0.35f) return;
        m_sample_timer = 0.0f;

#if defined(_WIN32)
        // CPU usage
        FILETIME sys_idle, sys_kernel, sys_user;
        FILETIME proc_create, proc_exit, proc_kernel, proc_user;

        if (GetSystemTimes(&sys_idle, &sys_kernel, &sys_user) &&
            GetProcessTimes(GetCurrentProcess(), &proc_create, &proc_exit, &proc_kernel, &proc_user))
        {
            const u64 sys_k = filetime_to_u64(sys_kernel);
            const u64 sys_u = filetime_to_u64(sys_user);
            const u64 proc_k = filetime_to_u64(proc_kernel);
            const u64 proc_u = filetime_to_u64(proc_user);

            if (m_prev_sys_kernel != 0)
            {
                const u64 sys_delta = (sys_k - m_prev_sys_kernel) + (sys_u - m_prev_sys_user);
                const u64 proc_delta = (proc_k - m_prev_proc_kernel) + (proc_u - m_prev_proc_user);

                if (sys_delta > 0) cpu_process_percent = static_cast<f32>(proc_delta * 100ull) / static_cast<f32>(sys_delta);
            }

            m_prev_sys_kernel = sys_k;
            m_prev_sys_user = sys_u;
            m_prev_proc_kernel = proc_k;
            m_prev_proc_user = proc_u;
        }

        // RAM usage
        PROCESS_MEMORY_COUNTERS_EX pmc {};
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        {
            ram_process_mb = static_cast<f32>(pmc.WorkingSetSize) / (1024.0f * 1024.0f);
            ram_commit_mb = static_cast<f32>(pmc.PrivateUsage) / (1024.0f * 1024.0f);
        }

        MEMORYSTATUSEX mem_status {};
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status))
        {
            ram_sys_total_gb = static_cast<f32>(mem_status.ullTotalPhys) / (1024.0f * 1024.0f * 1024.0f);
            ram_sys_used_gb = static_cast<f32>(mem_status.ullTotalPhys - mem_status.ullAvailPhys) / (1024.0f * 1024.0f * 1024.0f);
            ram_sys_percent = static_cast<u32>(mem_status.dwMemoryLoad);
        }
#endif
    }

private:

#if defined(_WIN32)
    static u64 filetime_to_u64(const FILETIME& ft) { return (static_cast<u64>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime; }
#endif

    f32 m_sample_timer = 0.0f;
    u64 m_prev_sys_kernel = 0;
    u64 m_prev_sys_user = 0;
    u64 m_prev_proc_kernel = 0;
    u64 m_prev_proc_user = 0;
};

struct FpsTracker
{
    f32 smoothed_fps = 0.0f;
    f32 smoothed_ms = 0.0f;
    f32 smoothed_cpu_ms = 0.0f;

    void update(f32 dt, f32 cpu_ms = 0.0f)
    {
        m_timer += dt;
        m_frame_count++;
        m_cpu_time_accum += cpu_ms;

        if (m_timer >= 0.5f)
        {
            smoothed_fps = static_cast<f32>(m_frame_count) / m_timer;
            smoothed_ms = (m_timer * 1000.0f) / static_cast<f32>(m_frame_count);
            smoothed_cpu_ms = m_cpu_time_accum / static_cast<f32>(m_frame_count);
            m_timer = 0.0f;
            m_frame_count = 0;
            m_cpu_time_accum = 0.0f;
        }
    }

private:

    f32 m_timer = 0.0f;
    u32 m_frame_count = 0;
    f32 m_cpu_time_accum = 0.0f;
};

// Context passed to widget functions
struct OverlayContext
{
    f32 dt = 0.0f;
    f32 smoothed_fps = 0.0f;
    f32 smoothed_ms = 0.0f;
    f32 cpu_time_ms = 0.0f;
    f32 gpu_time_ms = 0.0f;
    const SystemMetrics& metrics;
    WorldManager& world;
    Camera& camera;
    vulk::VulkanContext& vulkan;
    u32 chunks_drawn = 0;
    u32 chunks_culled = 0;
    bool& frustum_culling_enabled;
};

struct OverlayWidget
{
    std::string id;
    std::string title;
    std::function<void(OverlayContext&)> render_fn;
    bool enabled = true;
};

class DebugOverlay
{
public:

    u32 chunks_drawn = 0;
    u32 chunks_culled = 0;
    bool frustum_culling_enabled = true;

    DebugOverlay() { setup_default_widgets(); }

    void set_chunk_culling_stats(u32 drawn, u32 culled)
    {
        chunks_drawn = drawn;
        chunks_culled = culled;
    }

    void add_widget(std::string id, std::string title, std::function<void(OverlayContext&)> render_fn)
    {
        m_widgets.push_back(
            OverlayWidget {
                .id = std::move(id),
                .title = std::move(title),
                .render_fn = std::move(render_fn),
                .enabled = true,
            }
        );
    }

    void remove_widget(std::string_view id)
    {
        std::erase_if(m_widgets, [id](const OverlayWidget& w) { return w.id == id; });
    }

    void set_widget_enabled(std::string_view id, bool enabled)
    {
        for (auto& w : m_widgets)
        {
            if (w.id == id)
            {
                w.enabled = enabled;
                return;
            }
        }
    }

    void update(f32 dt, f32 cpu_ms = 0.0f)
    {
        m_fps.update(dt, cpu_ms);
        m_metrics.update(dt);
    }

    void draw(f32 dt, WorldManager& world, Camera& camera, vulk::VulkanContext& vulkan)
    {
        const f32 gpu_ms = vulkan.get_gpu_render_time_ms();
        OverlayContext ctx {
            .dt = dt,
            .smoothed_fps = m_fps.smoothed_fps,
            .smoothed_ms = m_fps.smoothed_ms,
            .cpu_time_ms = m_fps.smoothed_cpu_ms,
            .gpu_time_ms = gpu_ms,
            .metrics = m_metrics,
            .world = world,
            .camera = camera,
            .vulkan = vulkan,
            .chunks_drawn = chunks_drawn,
            .chunks_culled = chunks_culled,
            .frustum_culling_enabled = frustum_culling_enabled,
        };

        const f32 pad = 12.0f;
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const ImVec2 work_pos = viewport->WorkPos;
        ImGui::SetNextWindowPos(ImVec2(work_pos.x + pad, work_pos.y + pad), ImGuiCond_Always);
        ImGui::SetNextWindowBgAlpha(0.80f);

        constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                           ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                           ImGuiWindowFlags_NoNav;

        if (ImGui::Begin("##DebugOverlay", nullptr, flags))
        {
            for (usize i = 0; i < m_widgets.size(); ++i)
            {
                auto& widget = m_widgets[i];
                if (!widget.enabled || !widget.render_fn) continue;

                if (!widget.title.empty())
                {
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.70f, 0.75f, 0.85f, 1.0f), "%s", widget.title.c_str());
                }

                widget.render_fn(ctx);
            }
        }
        ImGui::End();
    }

private:

    void setup_default_widgets()
    {
        // FPS and timing
        add_widget("fps", "PERFORMANCE", [](OverlayContext& ctx) {
            const ImVec4 fps_color = (ctx.smoothed_fps >= 55.0f)   ? ImVec4(0.30f, 0.90f, 0.35f, 1.0f)
                                     : (ctx.smoothed_fps >= 30.0f) ? ImVec4(0.95f, 0.80f, 0.20f, 1.0f)
                                                                   : ImVec4(0.95f, 0.30f, 0.25f, 1.0f);

            ImGui::TextColored(fps_color, "%.1f FPS", static_cast<f64>(ctx.smoothed_fps));
            ImGui::SameLine();
            ImGui::TextDisabled("(%.2f ms/frame)", static_cast<f64>(ctx.smoothed_ms));

            // Budget relative to 60 FPS
            constexpr f32 target_60_ms = 1000.0f / 60.0f;
            const f32 gpu_60_pct = (ctx.gpu_time_ms / target_60_ms) * 100.0f;
            const f32 cpu_60_pct = (ctx.cpu_time_ms / target_60_ms) * 100.0f;

            const ImVec4 budget_gpu_color = (gpu_60_pct > 100.0f)   ? ImVec4(0.95f, 0.30f, 0.25f, 1.0f)
                                            : (gpu_60_pct >= 75.0f) ? ImVec4(0.95f, 0.80f, 0.20f, 1.0f)
                                                                    : ImVec4(0.30f, 0.90f, 0.35f, 1.0f);

            const ImVec4 budget_cpu_color = (cpu_60_pct > 100.0f)   ? ImVec4(0.95f, 0.30f, 0.25f, 1.0f)
                                            : (cpu_60_pct >= 75.0f) ? ImVec4(0.95f, 0.80f, 0.20f, 1.0f)
                                                                    : ImVec4(0.30f, 0.90f, 0.35f, 1.0f);

            if (ctx.gpu_time_ms > 0.001f)
                ImGui::Text("GPU Time: %.2f ms", static_cast<f64>(ctx.gpu_time_ms));
            else
                ImGui::TextDisabled("GPU Time: measuring...");

            ImGui::SameLine();
            ImGui::TextColored(budget_gpu_color, " (%.1f%%)", static_cast<f64>(gpu_60_pct));

            ImGui::Text("CPU Work: %.2f ms", static_cast<f64>(ctx.cpu_time_ms));

            ImGui::SameLine();
            ImGui::TextColored(budget_cpu_color, " (%.1f%%)", static_cast<f64>(cpu_60_pct));
        });

        // Hardware metrics: CPU, GPU, RAM
        add_widget("hardware", "SYSTEM USAGE", [](OverlayContext& ctx) {
            const auto gpu_stats = ctx.vulkan.get_memory_stats();
            const f64 vram_used_mb = static_cast<f64>(gpu_stats.used_bytes) / (1024.0 * 1024.0);
            const f64 vram_alloc_mb = static_cast<f64>(gpu_stats.allocated_bytes) / (1024.0 * 1024.0);
            const std::string_view gpu_name = ctx.vulkan.get_gpu_name();

            // CPU
            ImGui::Text("CPU (OS): %.1f%%", static_cast<f64>(ctx.metrics.cpu_process_percent));

            // RAM
            ImGui::Text(
                "RAM: %.1f MB (Commit: %.1f MB)", static_cast<f64>(ctx.metrics.ram_process_mb),
                static_cast<f64>(ctx.metrics.ram_commit_mb)
            );
            ImGui::TextDisabled(
                "Sys RAM: %u%% (%.1f / %.1f GB)", ctx.metrics.ram_sys_percent, static_cast<f64>(ctx.metrics.ram_sys_used_gb),
                static_cast<f64>(ctx.metrics.ram_sys_total_gb)
            );

            // GPU & VRAM
            if (!gpu_name.empty()) ImGui::Text("GPU: %.*s", static_cast<i32>(gpu_name.size()), gpu_name.data());
            ImGui::Text("VRAM: %.1f MB used (%.1f MB alloc)", vram_used_mb, vram_alloc_mb);
            if (gpu_stats.triangles_last_frame >= 1'000'000)
            {
                ImGui::TextDisabled(
                    "Draws: %u | Tris: %.2fM (%u)", gpu_stats.draw_calls_last_frame,
                    static_cast<f64>(gpu_stats.triangles_last_frame) / 1'000'000.0, gpu_stats.triangles_last_frame
                );
            }
            else
            {
                ImGui::TextDisabled("Draws: %u | Tris: %u", gpu_stats.draw_calls_last_frame, gpu_stats.triangles_last_frame);
            }
        });

        // World status
        add_widget("world", "WORLD", [](OverlayContext& ctx) {
            const u32 loaded_chunks = ctx.world.get_active_chunks_count();
            const u32 max_chunks = WorldManager::get_max_chunks_capacity();

            ImGui::Text("Chunks: %u drawn | %u culled", ctx.chunks_drawn, ctx.chunks_culled);
            ImGui::TextDisabled("Loaded: %u / %u", loaded_chunks, max_chunks);
            ImGui::TextDisabled("Far Plane: %.0f m", static_cast<f64>(ctx.camera.far_plane));
            ImGui::TextDisabled(
                "Position: (%.1f, %.1f, %.1f)", static_cast<f64>(ctx.camera.position.x), static_cast<f64>(ctx.camera.position.y),
                static_cast<f64>(ctx.camera.position.z)
            );
        });

        // Settings
        add_widget("settings", "SETTINGS", [](OverlayContext& ctx) {
            i32 current_rd = ctx.world.get_render_distance();
            ImGui::SetNextItemWidth(180.0f);
            if (ImGui::SliderInt("Render Dist", &current_rd, MIN_RENDER_DIST, MAX_RENDER_DIST, "%d chunks"))
            {
                ctx.world.set_render_distance(current_rd);
                ctx.camera.update_far_plane(current_rd, CHUNK_SIZE);
            }
            ImGui::Checkbox("Frustum Culling", &ctx.frustum_culling_enabled);

            bool vsync = ctx.vulkan.is_vsync_enabled();
            if (ImGui::Checkbox("VSync", &vsync)) ctx.vulkan.set_vsync(vsync);
        });
    }

    std::vector<OverlayWidget> m_widgets;
    FpsTracker m_fps;
    SystemMetrics m_metrics;
};

} // namespace vl
