#include "engine/engine.hpp"

#include <mimalloc-new-delete.h>

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