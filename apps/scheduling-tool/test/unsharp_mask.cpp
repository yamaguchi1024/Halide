// unsharp_mask.cpp
//
#include "Halide.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include <iostream>
#include <dlfcn.h>

using namespace Halide;

int main(int argc, char **argv) {
    if (!dlopen("libscheduling_tool.so", RTLD_LAZY)) {
        std::cerr << "Failed to load autoscheduler: " << dlerror() << "\n";
        return 1;
    }

    MachineParams params(32, 16000000, 40);
    Target target("x86-64-linux-sse41-avx-avx2");

    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("/home/yuka/Halide/apps/scheduling-tool/test/images/input.png");

    // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
    float sigma = 1.5f;

    // algorithm
    Var x("x"), y("y"), c("c");
    Func kernel("kernel");
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);

    Func bounded("bounded");
    bounded = BoundaryConditions::repeat_edge(input);

    Func gray("gray");
    gray(x, y) = 0.299f * bounded(x, y, 0) + 0.587f * bounded(x, y, 1) +
                 0.114f * bounded(x, y, 2);

    Func blur_y("blur_y");
    blur_y(x, y) = (kernel(0) * gray(x, y) +
                    kernel(1) * (gray(x, y-1) + gray(x, y+1)) +
                    kernel(2) * (gray(x, y-2) + gray(x, y+2)) +
                    kernel(3) * (gray(x, y-3) + gray(x, y+3)));

    Func blur("blur");
    blur(x, y) = (kernel(0) * blur_y(x, y) +
                    kernel(1) * (blur_y(x-1, y) + blur_y(x+1, y)) +
                    kernel(2) * (blur_y(x-2, y) + blur_y(x+2, y)) +
                    kernel(3) * (blur_y(x-3, y) + blur_y(x+3, y)));

    Func sharpen("sharpen");
    sharpen(x, y) = 2 * gray(x, y) - blur(x, y);

    Func ratio("ratio");
    ratio(x, y) = sharpen(x, y) / gray(x, y);

    Func unsharp("unsharp");
    unsharp(x, y, c) = ratio(x, y) * input(x, y, c);
    
    // end algorithm

    unsharp.set_estimate(x, 0, input.width()).set_estimate(y, 0, input.height()).set_estimate(c, 0, input.channels());
    Pipeline(unsharp).auto_schedule(target, params);

    Func output("output");
    output(x, y, c)  = cast<uint8_t>(unsharp(x, y, c));
    Buffer<uint8_t> buf(input.width(), input.height(), input.channels());
    double t = Halide::Tools::benchmark(3, 10, [&]() {
        output.realize(buf);
    });

    std::cerr << "Runtime from test: " << t*1000 << std::endl;
    Halide::Tools::save_image(buf, "unsharp_out.png");

    return 0;
}
