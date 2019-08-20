// 12.cpp
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

    // A scan in x followed by a downsample in y, with pointwise stuff in between
    const int N = 3;
    Buffer<float> a(1024, 1024);
    Func p1[N], p2[N], p3[N];
    Func s("scan");
    Var x, y;
    p1[0](x, y) = x + y;
    for (int i = 1; i < N; i++) {
        p1[i](x, y) = p1[i-1](x, y) + 1;
    }
    RDom r(1, 1023);
    s(x, y) = p1[N-1](x, y);
    s(r, y) += s(r-1, y);
    p2[0](x, y) = s(x, y);
    for (int i = 1; i < N; i++) {
        p2[i](x, y) = p2[i-1](x, y) + 1;
    }
    Func down("downsample");
    down(x, y) = p2[N-1](x, 2*y);
    p3[0](x, y) = down(x, y);
    for (int i = 1; i < N; i++) {
        p3[i](x, y) = p3[i-1](x, y) + 1;
    }

    p3[N-1].set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

    Pipeline(p3[N-1]).auto_schedule(target, params);

    return 0;
}
