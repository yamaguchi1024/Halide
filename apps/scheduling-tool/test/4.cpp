// 4.cpp
//
#include "Halide.h"
#include <dlfcn.h>

using namespace Halide;

int main(int argc, char **argv) {
    if (!dlopen("libscheduling_tool.so", RTLD_LAZY)) {
        std::cerr << "Failed to load autoscheduler: " << dlerror() << "\n";
        return 1;
    }

    MachineParams params(32, 16000000, 40);
    Target target("x86-64-linux-sse41-avx-avx2");

    Var x("x"), y("y");

    // Smaller footprint stencil -> smaller tiles
    Func f("f"), g("g"), h("h");
    f(x, y) = (x + y) * (x + 2*y) * (x + 3*y);
    h(x, y) = (f(x-1, y-1) + f(x, y-1) + f(x+1, y-1) +
            f(x-1, y  ) + f(x, y  ) + f(x+1, y  ) +
            f(x-1, y+1) + f(x, y+1) + f(x+1, y-1));

    h.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

    Pipeline(h).auto_schedule(target, params);

    return 0;
}
