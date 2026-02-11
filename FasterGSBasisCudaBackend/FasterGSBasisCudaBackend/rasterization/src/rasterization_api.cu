#include "rasterization_api.h"
#include "forward.h"
#include "backward.h"
#include "torch_utils.h"
#include "rasterization_config.h"
#include "helper_math.h"
#include <stdexcept>
#include <functional>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
faster_gs::rasterization::forward_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane)
{
    // all optimizable tensors must be passed as contiguous CUDA float tensors
    CHECK_INPUT(config::debug, means, "means");
    CHECK_INPUT(config::debug, scales, "scales");
    CHECK_INPUT(config::debug, rotations, "rotations");
    CHECK_INPUT(config::debug, opacities, "opacities");
    CHECK_INPUT(config::debug, sh_coefficients, "sh_coefficients");

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);

    auto [n_instances, instance_primitive_indices_selector] = forward(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane
    );

    return {
        image,
        primitive_buffers, tile_buffers, instance_buffers,
        n_instances, instance_primitive_indices_selector
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
faster_gs::rasterization::backward_wrapper(
    torch::Tensor& densification_info,
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& sh_coefficients,
    const torch::Tensor& primitive_buffers,
    const torch::Tensor& tile_buffers,
    const torch::Tensor& instance_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const int n_instances,
    const int instance_primitive_indices_selector)
{
    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_means = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_scales = torch::zeros({n_primitives, 3}, float_options);
    torch::Tensor grad_rotations = torch::zeros({n_primitives, 4}, float_options);
    torch::Tensor grad_opacities = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_sh_coefficients = torch::zeros({n_primitives, total_sh_bases, 3}, float_options);
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = torch::zeros({3, n_primitives}, float_options);
    torch::Tensor grad_color_helper = torch::zeros({3, n_primitives}, float_options);

    const bool update_densification_info = densification_info.size(0) > 0;

    backward(
        grad_image.data_ptr<float>(),
        image.data_ptr<float>(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(tile_buffers.data_ptr()),
        reinterpret_cast<char*>(instance_buffers.data_ptr()),
        reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
        reinterpret_cast<float*>(grad_opacities.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_coefficients.data_ptr<float>()),
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
        grad_conic_helper.data_ptr<float>(),
        grad_color_helper.data_ptr<float>(),
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,
        n_primitives,
        n_instances,
        instance_primitive_indices_selector,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y
    );

    return {grad_means, grad_scales, grad_rotations, grad_opacities, grad_sh_coefficients};
}
