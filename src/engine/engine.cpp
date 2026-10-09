#include "engine.hpp"

#include <print>

namespace vl
{

bool Engine::init()
{
    std::println("[Engine] Initialization...");
    return m_vulkan_context.init();
}

void Engine::run()
{
    std::println("[Engine] Main loop...");
}

void Engine::shutdown()
{
    std::println("[Engine] Shutting engine down...");
    m_vulkan_context.shutdown();
}

} // namespace vl