#pragma once

#include "helper_math.h"
#include <functional>

namespace faster_gs::rasterization {

    void backward(
        const float* grad_image,
        const float* image,
        const float3* spatial_means,
        const float* temporal_means,
        const float3* spatial_scales,
        const float* temporal_scales,
        const float4* left_isoclinic_rotations,
        const float4* right_isoclinic_rotations,
        const float* opacities,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        char* bucket_buffers_blob,
        float3* grad_spatial_means,
        float* grad_temporal_means,
        float3* grad_spatial_scales,
        float* grad_temporal_scales,
        float4* grad_left_isoclinic_rotations,
        float4* grad_right_isoclinic_rotations,
        float* grad_opacities,
        float3* grad_sh_coefficients_0,
        float3* grad_sh_coefficients_rest,
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
        const float timestamp);

}
