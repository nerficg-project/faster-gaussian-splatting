#pragma once

#include <torch/extension.h>

namespace faster_gs::adam {

    void adam_step_wrapper(
        const torch::Tensor& param_grad,
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const int step_count,
        const double learning_rate,
        const double beta1,
        const double beta2,
        const double epsilon);

}