#pragma once

#include "utils/types.hpp"

#include <algorithm>
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

    f32 move_speed = 5.5f;
    f32 sprint_multiplier = 2.5f;
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

} // namespace vl
