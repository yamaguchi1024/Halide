// 5.cpp
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

    // A stencil chain
    const int N = 8;
    Func f[N];
    f[0](x, y) = (x + y) * (x + 2*y) * (x + 3*y);
    for (int i = 1; i < N; i++) {
        Expr e = 0;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                e += f[i-1](x + dx, y + dy);
            }
        }
        f[i](x, y) = e;
    }
    f[N-1].set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

    Pipeline(f[N-1]).auto_schedule(target, params);

    return 0;
}
