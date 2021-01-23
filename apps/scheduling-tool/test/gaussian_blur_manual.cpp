// gaussian_blur.cpp
//
#include "Halide.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
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
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("/home/yuka/Halide/apps/scheduling-tool/test/images/input.png");

    float sigma = 1.5f;

    Var x("x"), y("y"), c("c");
    Func kernel("kernel");
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);

    Func in_bounded("in_bounded");
    in_bounded = BoundaryConditions::repeat_edge(input);
    Func blur_y("blur_y");
    blur_y(x, y, c) = (kernel(0) * in_bounded(x, y, c) +
            kernel(1) * (in_bounded(x, y-1, c) +
                in_bounded(x, y+1, c)) +
            kernel(2) * (in_bounded(x, y-2, c) +
                in_bounded(x, y+2, c)) +
            kernel(3) * (in_bounded(x, y-3, c) +
                in_bounded(x, y+3, c)));

    Func blur_x("blur_x");
    blur_x(x, y, c) = (kernel(0) * blur_y(x, y, c) +
            kernel(1) * (blur_y(x-1, y, c) +
                blur_y(x+1, y, c)) +
            kernel(2) * (blur_y(x-2, y, c) +
                blur_y(x+2, y, c)) +
            kernel(3) * (blur_y(x-3, y, c) +
                blur_y(x+3, y, c)));

    Func output("output");
    output(x, y, c)  = cast<uint8_t>(blur_x(x, y, c));

    // Scheduling!

    // Scheduling end

    output.print_loop_nest();

    Buffer<uint8_t> buf(input.width(), input.height(), input.channels());
    double t = Halide::Tools::benchmark(3, 10, [&]() {
            output.realize(buf);
        });

    std::cerr << "Runtime from test: " << t*1000 << std::endl;
    Halide::Tools::save_image(buf, "gaussian_blur_manual.png");

    return 0;
}
