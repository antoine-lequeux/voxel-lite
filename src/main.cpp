#include "engine/engine.hpp"

int main()
{
    vl::Engine engine;

    if (engine.init())
    {
        engine.run();
        engine.shutdown();
    }

    return 0;
}