#pragma once

#include "helper_math.h"
#include <functional>

namespace faster_gs::rasterization {

    void backward(
        const float* grad_image,
        const float* image,
        const float3* means,
        const float3* scales,
        const float4* rotations,
        const float3* sh_coefficients,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        float3* grad_means,
        float3* grad_scales,
        float4* grad_rotations,
        float* grad_opacities,
        float3* grad_sh_coefficients,
        float2* grad_mean2d_helper,
        float* grad_conic_helper,
        float* grad_color_helper,
        float* densification_info,
        const int n_primitives,
        const int n_instances,
        const int instance_primitive_indices_selector,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y);

}
