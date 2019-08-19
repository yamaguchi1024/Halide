// multiple_out.cpp
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
    // Use a fixed target for the analysis to get consistent results from this test.
    Target target("x86-64-linux-sse41-avx-avx2");

    Var x("x"), y("y");

    if (1) {
        // Long transpose chain.
        ImageParam im(Float(32), 2);
        Func f("f"), g("g"), h("h");

        f(x, y) = im(clamp(y*x, 0, 999), x);
        g(x, y) = f(clamp(y*x, 0, 999), x);
        h(x, y) = g(clamp(y*x, 0, 999), x);

        // Force everything to be compute root by accessing them in two separate outputs
        Func out1("out1"), out2("out2");
        out1(x, y) = f(x, y) + g(x, y) + h(x, y);
        out2(x, y) = f(x, y) + g(x, y) + h(x, y);

        out1.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);
        out2.set_estimate(x, 0, 1000).set_estimate(y, 0, 1000);
        Pipeline({out1, out2}).auto_schedule(target, params);

    }

    return 0;
}
