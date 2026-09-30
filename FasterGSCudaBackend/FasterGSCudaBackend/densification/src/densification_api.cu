#include "densification_api.h"
#include "mcmc.h"
#include "densification_config.h"
#include "helper_math.h"
#include "torch_utils.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>
#include <tuple>
#include <cstdint>

std::tuple<torch::Tensor, torch::Tensor>
faster_gs::densification::relocation_wrapper(
    const torch::Tensor& old_opacities,
    const torch::Tensor& old_scales,
    const torch::Tensor& n_samples_per_primitive)
{
    const int n_primitives = old_opacities.size(0);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor new_opacities = torch::empty({n_primitives, 1}, float_options);
    torch::Tensor new_scales = torch::empty({n_primitives, 3}, float_options);

    relocation_adjustment(
        old_opacities.contiguous().data_ptr<float>(),
        reinterpret_cast<float3*>(old_scales.contiguous().data_ptr<float>()),
        n_samples_per_primitive.contiguous().data_ptr<int64_t>(),
        new_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(new_scales.data_ptr<float>()),
        n_primitives
    );

    return std::make_tuple(new_opacities, new_scales);
}

void
faster_gs::densification::add_noise_wrapper(
    const torch::Tensor& raw_scales,
    const torch::Tensor& raw_rotations,
    const torch::Tensor& raw_opacities,
    torch::Tensor& means,
    const float current_lr)
{
    // tensors must be contiguous CUDA float tensors
    CHECK_INPUT(config::debug, raw_scales, "raw_scales");
    CHECK_INPUT(config::debug, raw_rotations, "raw_rotations");
    CHECK_INPUT(config::debug, raw_opacities, "raw_opacities");
    CHECK_INPUT(config::debug, means, "means");

    // draw philox seed and offset from torch's cuda generator
    auto* generator = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(generator->mutex_);
    const auto [seed, offset] = generator->philox_engine_inputs(4);

    add_noise(
        reinterpret_cast<float3*>(raw_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(raw_rotations.data_ptr<float>()),
        raw_opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        raw_scales.size(0),
        current_lr,
        seed,
        offset
    );

}
