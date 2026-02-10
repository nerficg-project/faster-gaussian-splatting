#pragma once

#include "helper_math.h"
#include <functional>

namespace faster_gs::rasterization {

    void backward(
        const float* grad_image,
        const float* image,
        float3* means,
        float3* scales,
        float4* rotations,
        float* opacities,
        float3* sh_coefficients_0,
        float3* sh_coefficients_rest,
        float2* moments_means,
        float2* moments_scales,
        float2* moments_rotations,
        float2* moments_opacities,
        float2* moments_sh_coefficients_0,
        float2* moments_sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        char* bucket_buffers_blob,
        float* grad_opacities,
        float3* grad_colors,
        float2* grad_mean2d_helper,
        float* grad_conic_helper,
        float* densification_info,
        const int n_primitives,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float current_mean_lr,
        const int adam_step_count);

}
