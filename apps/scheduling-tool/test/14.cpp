// 14.cpp
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

    // A schedule where it's insane to not compute inside an rvar
    Func f("f"), g("g");
    f(x, y) = x;
    f(x, y) += 1;

    RDom r(0, 100);
    g(x, y) = 0;
    g(x, y) += f(x, 1000*(y+r));

    g.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);

    Pipeline(g).auto_schedule(target, params);

    return 0;
}
