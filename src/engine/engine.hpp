#pragma once

#include "renderer/vulkancontext.hpp"

namespace vl
{

class Engine
{
public:

    bool init();
    void run();
    void shutdown();

private:

    vulk::VulkanContext m_vulkan_context;
};

} // namespace vl