// 10.cpp
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

    Func f_u8("f_u8");
    Func f_u64_1("f_u64_1");
    Func f_u64_2("f_u64_2");
    Buffer<float> a(1024 * 1024 + 2);

    f_u8(x) = (min(a(x) + 1, 17) * a(x+1) + a(x+2)) * a(x) * a(x) * a(x + 1) * a(x + 1);
    f_u64_1(x) = cast<float>(f_u8(x)) + 1;
    f_u64_2(x) = f_u64_1(x) * 3;

    // Ignoring the types, it would make sense to inline
    // everything into f_64_2 but this would vectorize fairly
    // narrowly, which is a waste of work for the first Func.

    f_u64_2.set_estimate(x, 0, 1024 * 1024);

    Pipeline(f_u64_2).auto_schedule(target, params);

    return 0;
}
