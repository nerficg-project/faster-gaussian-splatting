#include <torch/extension.h>
#include "rasterization_api.h"

namespace rasterization_api = faster_gs::rasterization;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
}
