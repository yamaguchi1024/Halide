// multiple_loops.cpp
// electron fails with failing to parse multiple_stages.update(1)! What to do with this?
//
#include "Halide.h"
#include <dlfcn.h>

using namespace Halide;

int main(int argc, char **argv) {
    if (!dlopen("libauto_schedule.so", RTLD_LAZY)) {
        std::cerr << "Failed to load autoscheduler: " << dlerror() << "\n";
        return 1;
    }

    MachineParams params(32, 16000000, 40);
    // Use a fixed target for the analysis to get consistent results from this test.
    Target target("x86-64-linux-sse41-avx-avx2");

    Var x("x"), y("y");

    // A Func with multiple stages, some of which include additional loops
    if (1) {
        Buffer<float> a(1024, 1024);
        Func f("multiple_stages"), g("g"), h("h");
        Var x, y;
        h(x, y) = pow(x, y);
        f(x, y) = a(x, y) * 2;
        f(x, y) += 17;
        RDom r(0, 10);
        f(x, y) += r * h(x, y);
        f(x, y) *= 2;
        f(0, y) = 23.0f;
        g(x, y) = f(x - 1, y - 1) + f(x + 1, y + 1);

        g.estimate(x, 1, 1022).estimate(y, 1, 1022);

        Pipeline(g).auto_schedule(target, params);
    }

    return 0;
}
