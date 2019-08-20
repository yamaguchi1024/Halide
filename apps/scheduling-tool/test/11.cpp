// 11.cpp
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

    Buffer<float> im_a(1024, 1024, "a"), im_b(1024, 1024, "b");
    im_a.fill(0.0f);
    im_b.fill(0.0f);

    Func c("c"), a("a"), b("b");
    Var i, j;
    a(j, i) = im_a(j, i);  // TODO: Add wrappers to the search space
    b(j, i) = im_b(j, i);
    RDom k(0, 1024);
    c(j, i) += a(k, i) * b(j, k);
    Func out("out");
    out(j, i) = c(j, i);

    out.set_estimate(j, 0, 1024).set_estimate(i, 0, 1024);

    Pipeline(out).auto_schedule(target, params);

    return 0;
}
