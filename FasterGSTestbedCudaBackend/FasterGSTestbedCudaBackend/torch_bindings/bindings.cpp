#include <torch/extension.h>
#include "rasterization_api.h"
#include "adam.h"

namespace rasterization_api = faster_gs::rasterization;
namespace adam_api = faster_gs::adam;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("adam_step", &adam_api::adam_step_wrapper);
}
