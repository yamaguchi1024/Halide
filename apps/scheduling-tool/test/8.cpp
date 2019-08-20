// 8.cpp
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

    // A Func with multiple stages, some of which include additional loops
    Buffer<float> a(1024, 1024);
    Func f("multiple_stages"), g("g"), h("h");
    h(x, y) = pow(x, y);
    f(x, y) = a(x, y) * 2;
    f(x, y) += 17;
    RDom r(0, 10);
    f(x, y) += r * h(x, y);
    f(x, y) *= 2;
    f(0, y) = 23.0f;
    g(x, y) = f(x - 1, y - 1) + f(x + 1, y + 1);

    g.set_estimate(x, 1, 1022).set_estimate(y, 1, 1022);

    Pipeline(g).auto_schedule(target, params);

    return 0;
}
