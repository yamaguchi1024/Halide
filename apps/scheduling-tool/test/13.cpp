// 13.cpp
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

    // A gather that only uses a small portion of a potentially
    // large LUT. The number of points computed should be less
    // than points computed minimum, and the LUT should be
    // inlined, even if it's really expensive.
    Func lut("lut");
    Var x;
    lut(x) = (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6);

    Func idx("idx");
    idx(x) = x * (10000 - x);

    Func out("out");
    out(x) = lut(clamp(idx(x), 0, 100000));

    out.set_estimate(x, 0, 10);

    Pipeline(out).auto_schedule(target, params);

    return 0;
}
