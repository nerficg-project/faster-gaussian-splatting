#pragma once

#include <torch/extension.h>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
    forward_wrapper(
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& opacities,
        const torch::Tensor& sh_coefficients,
        const torch::Tensor& sh_coefficients_0,
        const torch::Tensor& sh_coefficients_rest,
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
        const bool use_tight_bounds,
        const bool use_opacity_based_bounds,
        const bool use_fused_activations,
        const bool use_load_balanced_instance_creation,
        const bool use_tile_based_culling,
        const bool use_per_gaussian_backward,
        const bool use_separate_sorting);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
        torch::Tensor& densification_info,
        const torch::Tensor& grad_image,
        const torch::Tensor& image,
        const torch::Tensor& means,
        const torch::Tensor& scales,
        const torch::Tensor& rotations,
        const torch::Tensor& sh_coefficients,
        const torch::Tensor& sh_coefficients_rest,
        const torch::Tensor& primitive_buffers,
        const torch::Tensor& tile_buffers,
        const torch::Tensor& instance_buffers,
        const torch::Tensor& bucket_buffers,
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
        const bool use_tight_bounds,
        const bool use_opacity_based_bounds,
        const bool use_fused_activations,
        const bool use_load_balanced_instance_creation,
        const bool use_tile_based_culling,
        const bool use_per_gaussian_backward,
        const bool use_separate_sorting,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector);

}
