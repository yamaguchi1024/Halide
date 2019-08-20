// 6.cpp
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

    // An outer product
    Buffer<float> a(2048), b(2048);
    Func f;
    f(x, y) = a(x) * b(y);

    f.set_estimate(x, 0, 2048).set_estimate(y, 0, 2048);

    Pipeline(f).auto_schedule(target, params);

    return 0;
}
