#pragma once

#include "utils/types.hpp"

#include <algorithm>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vl
{

class Camera
{
public:

    glm::vec3 position {0.0f, 1.25f, 3.5f};
    f32 yaw_degrees = -90.0f;
    f32 pitch_degrees = -16.0f;

    f32 fov_degrees = 70.0f;
    f32 near_plane = 0.05f;
    f32 far_plane = 1000.0f;

    void update_far_plane(i32 render_distance, i32 chunk_size)
    {
        const f32 needed = static_cast<f32>(render_distance + 2) * static_cast<f32>(chunk_size) * 1.45f + 64.0f;
        far_plane = std::max(1000.0f, needed);
    }

    f32 move_speed = 10.0f;
    f32 sprint_multiplier = 15.0f;
    f32 mouse_sensitivity = 0.12f;

    [[nodiscard]] glm::vec3 forward() const
    {
        const f32 yaw_rad = glm::radians(yaw_degrees);
        const f32 pitch_rad = glm::radians(pitch_degrees);

        glm::vec3 dir {};
        dir.x = std::cos(yaw_rad) * std::cos(pitch_rad);
        dir.y = std::sin(pitch_rad);
        dir.z = std::sin(yaw_rad) * std::cos(pitch_rad);
        return glm::normalize(dir);
    }

    [[nodiscard]] glm::vec3 flat_forward() const
    {
        const f32 yaw_rad = glm::radians(yaw_degrees);
        return glm::normalize(glm::vec3 {std::cos(yaw_rad), 0.0f, std::sin(yaw_rad)});
    }

    [[nodiscard]] glm::vec3 right() const
    {
        constexpr glm::vec3 world_up {0.0f, 1.0f, 0.0f};
        return glm::normalize(glm::cross(forward(), world_up));
    }

    [[nodiscard]] glm::vec3 up() const { return glm::normalize(glm::cross(right(), forward())); }

    void rotate(f32 delta_yaw_degrees, f32 delta_pitch_degrees)
    {
        yaw_degrees += delta_yaw_degrees;
        pitch_degrees = std::clamp(pitch_degrees + delta_pitch_degrees, -89.0f, 89.0f);
    }

    [[nodiscard]] glm::mat4 view_matrix() const
    {
        constexpr glm::vec3 world_up {0.0f, 1.0f, 0.0f};
        return glm::lookAt(position, position + forward(), world_up);
    }

    [[nodiscard]] glm::mat4 projection_matrix(f32 aspect_ratio) const
    {
        const f32 safe_aspect = (aspect_ratio > 0.001f) ? aspect_ratio : (16.0f / 9.0f);
        glm::mat4 proj = glm::perspective(glm::radians(fov_degrees), safe_aspect, near_plane, far_plane);
        proj[1][1] *= -1.0f; // Flip Y for Vulkan NDC coordinates
        return proj;
    }

    [[nodiscard]] glm::mat4 view_projection_matrix(f32 aspect_ratio) const
    {
        return projection_matrix(aspect_ratio) * view_matrix();
    }
};

struct Frustum
{
    std::array<glm::vec4, 6> planes {}; // Ax + By + Cz + D = 0

    [[nodiscard]] static Frustum from_view_projection(const glm::mat4& vp)
    {
        Frustum f {};
        const glm::vec4 row0 {vp[0][0], vp[1][0], vp[2][0], vp[3][0]};
        const glm::vec4 row1 {vp[0][1], vp[1][1], vp[2][1], vp[3][1]};
        const glm::vec4 row2 {vp[0][2], vp[1][2], vp[2][2], vp[3][2]};
        const glm::vec4 row3 {vp[0][3], vp[1][3], vp[2][3], vp[3][3]};

        f.planes[0] = row3 + row0; // Left
        f.planes[1] = row3 - row0; // Right
        f.planes[2] = row3 + row1; // Bottom
        f.planes[3] = row3 - row1; // Top
        f.planes[4] = row2;        // Near
        f.planes[5] = row3 - row2; // Far

        for (glm::vec4& p : f.planes)
        {
            const f32 len = glm::length(glm::vec3(p));
            if (len > 0.00001f) p /= len;
        }

        return f;
    }
};

} // namespace vl
