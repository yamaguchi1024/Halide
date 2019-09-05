// unsharp_mask.cpp
//
#include "Halide.h"
#include "halide_image_io.h"
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

    Halide::Buffer<uint8_t> in = Halide::Tools::load_image("/home/yuka/Halide/apps/scheduling-tool/test/images/input.png");

    // Define a 7x7 Gaussian Blur with a repeat-edge boundary condition.
    float sigma = 1.5f;

    Var x, y, c;
    Func kernel;
    kernel(x) = exp(-x*x/(2*sigma*sigma)) / (sqrtf(2*M_PI)*sigma);

    Func in_bounded = BoundaryConditions::repeat_edge(in);

    Func gray;
    gray(x, y) = max(in_bounded(x, y, 0),
                     max(in_bounded(x, y, 1),
                         in_bounded(x, y, 2)));

    Func blur_y;
    blur_y(x, y) = (kernel(0) * gray(x, y) +
                    kernel(1) * (gray(x, y-1) +
                                 gray(x, y+1)) +
                    kernel(2) * (gray(x, y-2) +
                                 gray(x, y+2)) +
                    kernel(3) * (gray(x, y-3) +
                                 gray(x, y+3)));

    Func blur_x;
    blur_x(x, y) = (kernel(0) * blur_y(x, y) +
                    kernel(1) * (blur_y(x-1, y) +
                                 blur_y(x+1, y)) +
                    kernel(2) * (blur_y(x-2, y) +
                                 blur_y(x+2, y)) +
                    kernel(3) * (blur_y(x-3, y) +
                                 blur_y(x+3, y)));

    Func sharpen;
    sharpen(x, y) = 2 * gray(x, y) - blur_x(x, y);

    Func ratio;
    ratio(x, y) = sharpen(x, y) / gray(x, y);

    Func result;
    result(x, y, c) = cast<float>(ratio(x, y) * in(x, y, c));

    result.set_estimate(x, 0, in.width()).set_estimate(y, 0, in.height()).set_estimate(c, 0, in.channels());
    Pipeline(result).auto_schedule(target, params);

    return 0;
}
