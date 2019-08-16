#include "Halide.h"

#include <dlfcn.h>
//#include <Halide/tools/schedule_yourself.h>

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

    if (1) {
        // In a pipeline with huge expensive stencils and low memory costs, nothing should be fused
        Func f("f"), g("g"), h("h");
        f(x, y) = (x + y) * (x + 2*y) * (x + 3*y) * (x + 4*y) * (x + 5*y);
        Expr e = 0;
        for (int i = 0; i < 100; i++) {
            e += f(x + i*10, y + i*10);
        }
        g(x, y) = e;
        e = 0;
        for (int i = 0; i < 100; i++) {
            e += g(x + i*10, y + i*10);
        }
        h(x, y) = e;

        h.estimate(x, 0, 1000).estimate(y, 0, 1000);

        Pipeline(h).auto_schedule(target, params);
        //Pipeline(h).schedule_yourself(target, params);
        //schedule_yourself(Pipeline(h), target, params);

        /*
        h.realize(400, 400);
        h.realize(400, 400);
        h.realize(400, 400);
        h.realize(400, 400);
        h.realize(400, 400);
        */
    }

    return 0;
}
