// 7.cpp
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

    // A separable downsample that models the start of local_laplacian
    Buffer<float> in(2048, 2048);
    Var k;
    Func orig("orig"), expensive("expensive"), downy("downy"), downx("downx");
    Expr e = 0;
    for (int i = 0; i < 100; i++) {
        e += 1;
        e *= e;
    }
    orig(x, y) = e;
    expensive(x, y, k) = orig(x, y) * orig(x, y) + (x + orig(x, y)) * (1 + orig(x, y)) + sqrt(k + orig(x, y));
    downy(x, y, k) = expensive(x, 2*y - 1, k) + expensive(x, 2*y, k) + expensive(x, 2*y+1, k) + expensive(x, 2*y + 2, k);
    downx(x, y, k) = downy(2*x-1, y, k) + downy(2*x, y, k) + downy(2*x + 1, y, k) + downy(2*x + 2, y, k);
    downx.set_estimate(x, 1, 1022).set_estimate(y, 1, 1022).set_estimate(k, 0, 256);

    Pipeline(downx).auto_schedule(target, params);

    return 0;
}
