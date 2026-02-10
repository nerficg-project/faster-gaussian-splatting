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
        const float current_mean_lr,
        const int adam_step_count);

    void backward_wrapper(
        torch::Tensor& densification_info,
        torch::Tensor& means,
        torch::Tensor& scales,
        torch::Tensor& rotations,
        torch::Tensor& opacities,
        torch::Tensor& sh_coefficients_0,
        torch::Tensor& sh_coefficients_rest,
        torch::Tensor& moments_means,
        torch::Tensor& moments_scales,
        torch::Tensor& moments_rotations,
        torch::Tensor& moments_opacities,
        torch::Tensor& moments_sh_coefficients_0,
        torch::Tensor& moments_sh_coefficients_rest,
        const torch::Tensor& grad_image,
        const torch::Tensor& image,
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
        const float current_mean_lr, // fuse this in as well based on adam_step_count
        const int adam_step_count,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector);

}
