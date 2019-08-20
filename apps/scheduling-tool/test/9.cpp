// 9.cpp
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

    // A scan with pointwise stages before and after
    Buffer<float> a(1024, 1024);
    Func before[5];
    Func after[5];
    Func s("scan");
    before[0](x, y) = x + y;
    for (int i = 1; i < 5; i++) {
        before[i](x, y) = before[i-1](x, y) + 1;
    }
    RDom r(1, 1023);
    s(x, y) = before[4](x, y);
    s(r, y) += s(r-1, y);
    after[0](x, y) = s(y, x) + s(y, x+100);
    for (int i = 1; i < 5; i++) {
        after[i](x, y) = after[i-1](x, y) + 1;
    }

    after[4].set_estimate(x, 0, 1024).set_estimate(y, 0, 1024);

    Pipeline(after[4]).auto_schedule(target, params);

    return 0;
}
