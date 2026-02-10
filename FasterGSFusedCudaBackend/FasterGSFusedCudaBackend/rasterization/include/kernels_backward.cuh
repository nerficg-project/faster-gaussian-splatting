#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "sh_utils.cuh"
#include "buffer_utils.h"
#include "helper_math.h"
#include "utils.h"
#include <cstdint>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace faster_gs::rasterization::kernels::backward {

    __global__ void preprocess_backward_cu(
        float3* __restrict__ means,
        float3* __restrict__ scales,
        float4* __restrict__ rotations,
        float* __restrict__ opacities,
        float3* __restrict__ sh_coefficients_0,
        float3* __restrict__ sh_coefficients_rest,
        float2* __restrict__ moments_means,
        float2* __restrict__ moments_scales,
        float2* __restrict__ moments_rotations,
        float2* __restrict__ moments_opacities,
        float2* __restrict__ moments_sh_coefficients_0,
        float2* __restrict__ moments_sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        const float* __restrict__ grad_opacities,
        const float3* __restrict__ grad_colors,
        float* __restrict__ densification_info,
        const uint n_primitives,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float step_size_means,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        const float step_size_opacities = config::lr_opacities * bias_correction1_rcp;
        adam_step_helper<1, 0>(grad_opacities[primitive_idx], opacities, moments_opacities, primitive_idx, step_size_opacities, bias_correction2_sqrt_rcp);

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_0, sh_coefficients_rest,
            moments_sh_coefficients_0, moments_sh_coefficients_rest,
            grad_colors,
            mean3d, cam_position[0], primitive_idx,
            active_sh_bases, total_sh_bases,
            bias_correction1_rcp, bias_correction2_sqrt_rcp
        );

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // compute 3d covariance from scale and rotation
        const float3 raw_scale = scales[primitive_idx];
        const float3 variance = expf(2.0f * raw_scale);
        const float4 raw_rotation = rotations[primitive_idx];
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq);
        const mat3x3 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
        };
        const mat3x3_triu cov3d {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
        };

        // ewa splatting gradient helpers
        const float clip_left = (-0.15f * width - center_x) / focal_x;
        const float clip_right = (1.15f * width - center_x) / focal_x;
        const float clip_top = (-0.15f * height - center_y) / focal_y;
        const float clip_bottom = (1.15f * height - center_y) / focal_y;
        const float x_clipped = clamp(x, clip_left, clip_right);
        const float y_clipped = clamp(y, clip_top, clip_bottom);
        const float j11 = focal_x / depth;
        const float j13 = -j11 * x_clipped;
        const float j22 = focal_y / depth;
        const float j23 = -j22 * y_clipped;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z
        );
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z
        );
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
        );
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
        );

        // 2d covariance gradient
        const float a = dot(jwc_r1, jw_r1) + config::dilation, b = dot(jwc_r1, jw_r2), c = dot(jwc_r2, jw_r2) + config::dilation;
        const float aa = a * a, bb = b * b, cc = c * c;
        const float ac = a * c, ab = a * b, bc = b * c;
        const float determinant = ac - bb;
        const float determinant_sq = determinant * determinant;
        const float determinant_rcp_sq = 1.0f / determinant_sq;
        const float3 dL_dconic = make_float3(
            grad_conic[primitive_idx],
            grad_conic[n_primitives + primitive_idx],
            grad_conic[2 * n_primitives + primitive_idx]
        );
        const float3 dL_dcov2d = determinant_rcp_sq * make_float3(
            2.0f * bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
            bc * dL_dconic.x - (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
            2.0f * ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z
        );

        // 3d covariance gradient
        const mat3x3_triu dL_dcov3d = {
            jw_r1.x * jw_r1.x * dL_dcov2d.x + 2.0f * jw_r1.x * jw_r2.x * dL_dcov2d.y + jw_r2.x * jw_r2.x * dL_dcov2d.z,
            jw_r1.x * jw_r1.y * dL_dcov2d.x + (jw_r1.x * jw_r2.y + jw_r1.y * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.y * dL_dcov2d.z,
            jw_r1.x * jw_r1.z * dL_dcov2d.x + (jw_r1.x * jw_r2.z + jw_r1.z * jw_r2.x) * dL_dcov2d.y + jw_r2.x * jw_r2.z * dL_dcov2d.z,
            jw_r1.y * jw_r1.y * dL_dcov2d.x + 2.0f * jw_r1.y * jw_r2.y * dL_dcov2d.y + jw_r2.y * jw_r2.y * dL_dcov2d.z,
            jw_r1.y * jw_r1.z * dL_dcov2d.x + (jw_r1.y * jw_r2.z + jw_r1.z * jw_r2.y) * dL_dcov2d.y + jw_r2.y * jw_r2.z * dL_dcov2d.z,
            jw_r1.z * jw_r1.z * dL_dcov2d.x + 2.0f * jw_r1.z * jw_r2.z * dL_dcov2d.y + jw_r2.z * jw_r2.z * dL_dcov2d.z,
        };

        // gradient of J * W
        const float3 dL_djw_r1 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.x + jwc_r2.x * dL_dcov2d.y,
            jwc_r1.y * dL_dcov2d.x + jwc_r2.y * dL_dcov2d.y,
            jwc_r1.z * dL_dcov2d.x + jwc_r2.z * dL_dcov2d.y
        );
        const float3 dL_djw_r2 = 2.0f * make_float3(
            jwc_r1.x * dL_dcov2d.y + jwc_r2.x * dL_dcov2d.z,
            jwc_r1.y * dL_dcov2d.y + jwc_r2.y * dL_dcov2d.z,
            jwc_r1.z * dL_dcov2d.y + jwc_r2.z * dL_dcov2d.z
        );

        // gradient of non-zero entries in J
        const float dL_dj11 = w2c_r1.x * dL_djw_r1.x + w2c_r1.y * dL_djw_r1.y + w2c_r1.z * dL_djw_r1.z;
        const float dL_dj22 = w2c_r2.x * dL_djw_r2.x + w2c_r2.y * dL_djw_r2.y + w2c_r2.z * dL_djw_r2.z;
        const float dL_dj13 = w2c_r3.x * dL_djw_r1.x + w2c_r3.y * dL_djw_r1.y + w2c_r3.z * dL_djw_r1.z;
        const float dL_dj23 = w2c_r3.x * dL_djw_r2.x + w2c_r3.y * dL_djw_r2.y + w2c_r3.z * dL_djw_r2.z;

        // load gradient of 2d mean
        const float2 dL_dmean2d = grad_mean2d[primitive_idx];

        // for adaptive density control
        if (densification_info != nullptr) {
            densification_info[primitive_idx] += 1.0f;
            const float2 dL_dmean2d_ndc = 0.5f * make_float2(
                dL_dmean2d.x * width,
                dL_dmean2d.y * height
            );
            densification_info[n_primitives + primitive_idx] += length(dL_dmean2d_ndc);
        }

        // mean3d camera space gradient from mean2d
        float3 dL_dmean3d_cam = make_float3(
            j11 * dL_dmean2d.x,
            j22 * dL_dmean2d.y,
            -j11 * x * dL_dmean2d.x - j22 * y * dL_dmean2d.y
        );

        // add mean3d camera space gradient from J while accounting for clipping
        const bool valid_x = x >= clip_left && x <= clip_right;
        const bool valid_y = y >= clip_top && y <= clip_bottom;
        if (valid_x) dL_dmean3d_cam.x -= j11 * dL_dj13 / depth;
        if (valid_y) dL_dmean3d_cam.y -= j22 * dL_dj23 / depth;
        const float factor_x = 1.0f + static_cast<float>(valid_x);
        const float factor_y = 1.0f + static_cast<float>(valid_y);
        dL_dmean3d_cam.z += (j11 * (factor_x * x_clipped * dL_dj13 - dL_dj11) + j22 * (factor_y * y_clipped * dL_dj23 - dL_dj22)) / depth;

        // 3d mean gradient from splatting
        const float3 dL_dmean3d_from_splatting = make_float3(
            w2c_r1.x * dL_dmean3d_cam.x + w2c_r2.x * dL_dmean3d_cam.y + w2c_r3.x * dL_dmean3d_cam.z,
            w2c_r1.y * dL_dmean3d_cam.x + w2c_r2.y * dL_dmean3d_cam.y + w2c_r3.y * dL_dmean3d_cam.z,
            w2c_r1.z * dL_dmean3d_cam.x + w2c_r2.z * dL_dmean3d_cam.y + w2c_r3.z * dL_dmean3d_cam.z
        );

        // meand3d gradient
        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        adam_step_helper<3, 0>(dL_dmean3d.x, reinterpret_cast<float*>(means), moments_means, primitive_idx, step_size_means, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 1>(dL_dmean3d.y, reinterpret_cast<float*>(means), moments_means, primitive_idx, step_size_means, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 2>(dL_dmean3d.z, reinterpret_cast<float*>(means), moments_means, primitive_idx, step_size_means, bias_correction2_sqrt_rcp);

        // scale gradient
        const float3 dL_dvariance = make_float3(
            R.m11 * R.m11 * dL_dcov3d.m11 + R.m21 * R.m21 * dL_dcov3d.m22 + R.m31 * R.m31 * dL_dcov3d.m33 +
                2.0f * (R.m11 * R.m21 * dL_dcov3d.m12 + R.m11 * R.m31 * dL_dcov3d.m13 + R.m21 * R.m31 * dL_dcov3d.m23),
            R.m12 * R.m12 * dL_dcov3d.m11 + R.m22 * R.m22 * dL_dcov3d.m22 + R.m32 * R.m32 * dL_dcov3d.m33 +
                2.0f * (R.m12 * R.m22 * dL_dcov3d.m12 + R.m12 * R.m32 * dL_dcov3d.m13 + R.m22 * R.m32 * dL_dcov3d.m23),
            R.m13 * R.m13 * dL_dcov3d.m11 + R.m23 * R.m23 * dL_dcov3d.m22 + R.m33 * R.m33 * dL_dcov3d.m33 +
                2.0f * (R.m13 * R.m23 * dL_dcov3d.m12 + R.m13 * R.m33 * dL_dcov3d.m13 + R.m23 * R.m33 * dL_dcov3d.m23)
        );
        const float3 dL_dscale = 2.0f * variance * dL_dvariance;
        const float step_size_scales = config::lr_scales * bias_correction1_rcp;
        adam_step_helper<3, 0>(dL_dscale.x, reinterpret_cast<float*>(scales), moments_scales, primitive_idx, step_size_scales, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 1>(dL_dscale.y, reinterpret_cast<float*>(scales), moments_scales, primitive_idx, step_size_scales, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 2>(dL_dscale.z, reinterpret_cast<float*>(scales), moments_scales, primitive_idx, step_size_scales, bias_correction2_sqrt_rcp);

        // rotation gradient
        const mat3x3 dL_dR = {
            2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13),
            2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13),
            2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13),
            2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23),
            2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23),
            2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23),
            2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33),
            2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33),
            2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33)
        };
        const float4 dL_drotation = convert_quaternion_to_rotation_matrix_backward(raw_rotation, dL_dR);
        const float step_size_rotations = config::lr_rotations * bias_correction1_rcp;
        adam_step_helper<4, 0>(dL_drotation.x, reinterpret_cast<float*>(rotations), moments_rotations, primitive_idx, step_size_rotations, bias_correction2_sqrt_rcp);
        adam_step_helper<4, 1>(dL_drotation.y, reinterpret_cast<float*>(rotations), moments_rotations, primitive_idx, step_size_rotations, bias_correction2_sqrt_rcp);
        adam_step_helper<4, 2>(dL_drotation.z, reinterpret_cast<float*>(rotations), moments_rotations, primitive_idx, step_size_rotations, bias_correction2_sqrt_rcp);
        adam_step_helper<4, 3>(dL_drotation.w, reinterpret_cast<float*>(rotations), moments_rotations, primitive_idx, step_size_rotations, bias_correction2_sqrt_rcp);

    }

    // based on https://github.com/humansensinglab/taming-3dgs/blob/fd0f7d9edfe135eb4eefd3be82ee56dada7f2a16/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu#L404
    __global__ void blend_backward_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_bucket_offsets,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        const float* __restrict__ grad_image,
        const float* __restrict__ image,
        const float* __restrict__ tile_final_transmittances,
        const uint* __restrict__ tile_max_n_processed,
        const uint* __restrict__ tile_n_processed,
        const uint* __restrict__ bucket_tile_index,
        const float4* __restrict__ bucket_color_transmittance,
        float2* __restrict__ grad_mean2d,
        float* __restrict__ grad_conic,
        float* __restrict__ grad_opacity,
        float3* __restrict__ grad_colors,
        const uint n_primitives,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        const uint bucket_idx = block.group_index().x;
        const uint lane_idx = warp.thread_rank();

        const uint tile_idx = bucket_tile_index[bucket_idx];
        const uint2 tile_instance_range = tile_instance_ranges[tile_idx];
        const int tile_n_primitives = tile_instance_range.y - tile_instance_range.x;
        const uint tile_first_bucket_offset = (tile_idx == 0) ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int tile_bucket_idx = bucket_idx - tile_first_bucket_offset;
        if (tile_bucket_idx * 32 >= tile_max_n_processed[tile_idx]) return;

        const int tile_primitive_idx = tile_bucket_idx * 32 + lane_idx;
        const int instance_idx = tile_instance_range.x + tile_primitive_idx;
        const bool valid_primitive = tile_primitive_idx < tile_n_primitives;

        // load gaussian data
        uint primitive_idx = 0;
        float2 mean2d = {0.0f, 0.0f};
        float3 conic = {0.0f, 0.0f, 0.0f};
        float opacity = 0.0f;
        float3 color = {0.0f, 0.0f, 0.0f};
        float3 color_grad_factor = {0.0f, 0.0f, 0.0f};
        if (valid_primitive) {
            primitive_idx = instance_primitive_indices[instance_idx];
            mean2d = primitive_mean2d[primitive_idx];
            const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
            conic = make_float3(conic_opacity);
            opacity = conic_opacity.w;
            const float3 color_unclamped = primitive_color[primitive_idx];
            color = fmaxf(color_unclamped, 0.0f);
            if (color_unclamped.x >= 0.0f) color_grad_factor.x = 1.0f;
            if (color_unclamped.y >= 0.0f) color_grad_factor.y = 1.0f;
            if (color_unclamped.z >= 0.0f) color_grad_factor.z = 1.0f;
        }

        // helpers
        const float3 background = bg_color[0];
        const uint n_pixels = width * height;

        // gradient accumulation
        float2 dL_dmean2d_accum = {0.0f, 0.0f};
        float3 dL_dconic_accum = {0.0f, 0.0f, 0.0f};
        float dL_dopacity_accum = 0.0f;
        float3 dL_dcolor_accum = {0.0f, 0.0f, 0.0f};

        // tile metadata
        const uint2 tile_coords = {tile_idx % grid_width, tile_idx / grid_width};
        const uint2 start_pixel_coords = {tile_coords.x * config::tile_width, tile_coords.y * config::tile_height};

        uint last_contributor;
        float3 color_pixel_after;
        float transmittance;
        float3 grad_color_pixel;
        float grad_alpha_common;

        bucket_color_transmittance += bucket_idx * config::block_size_blend;
        __shared__ uint collected_last_contributor[32];
        __shared__ float4 collected_color_pixel_after_transmittance[32];
        __shared__ float4 collected_grad_info_pixel[32];

        // iterate over all pixels in the tile
        #pragma unroll
        for (int i = 0; i < config::block_size_blend + 31; ++i) {
            if (i % 32 == 0) {
                const uint local_idx = i + lane_idx;
                if (local_idx < config::block_size_blend) {
                    const float4 color_transmittance = bucket_color_transmittance[local_idx];
                    const uint2 pixel_coords = {start_pixel_coords.x + local_idx % config::tile_width, start_pixel_coords.y + local_idx / config::tile_width};
                    const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
                    // final values from forward pass before background blend and the respective gradients
                    float3 color_pixel_w_bg, grad_color_pixel;
                    if (pixel_coords.x < width && pixel_coords.y < height) {
                        color_pixel_w_bg = make_float3(
                            image[pixel_idx],
                            image[n_pixels + pixel_idx],
                            image[2 * n_pixels + pixel_idx]
                        );
                        grad_color_pixel = make_float3(
                            grad_image[pixel_idx],
                            grad_image[n_pixels + pixel_idx],
                            grad_image[2 * n_pixels + pixel_idx]
                        );
                    }
                    const float final_transmittance = tile_final_transmittances[pixel_idx];
                    collected_color_pixel_after_transmittance[lane_idx] = make_float4(
                        color_pixel_w_bg - final_transmittance * background - make_float3(color_transmittance),
                        color_transmittance.w
                    );
                    collected_grad_info_pixel[lane_idx] = make_float4(
                        grad_color_pixel,
                        final_transmittance * -dot(grad_color_pixel, background)
                    );
                    collected_last_contributor[lane_idx] = tile_n_processed[pixel_idx];
                }
                warp.sync();
            }

            if (i > 0) {
                last_contributor = warp.shfl_up(last_contributor, 1);
                color_pixel_after.x = warp.shfl_up(color_pixel_after.x, 1);
                color_pixel_after.y = warp.shfl_up(color_pixel_after.y, 1);
                color_pixel_after.z = warp.shfl_up(color_pixel_after.z, 1);
                transmittance = warp.shfl_up(transmittance, 1);
                grad_color_pixel.x = warp.shfl_up(grad_color_pixel.x, 1);
                grad_color_pixel.y = warp.shfl_up(grad_color_pixel.y, 1);
                grad_color_pixel.z = warp.shfl_up(grad_color_pixel.z, 1);
                grad_alpha_common = warp.shfl_up(grad_alpha_common, 1);
            }

            // which pixel index should this thread deal with?
            const int idx = i - static_cast<int>(lane_idx);
            const uint2 pixel_coords = {start_pixel_coords.x + idx % config::tile_width, start_pixel_coords.y + idx / config::tile_width};
            const bool valid_pixel = pixel_coords.x < width && pixel_coords.y < height;

            // leader thread loads values from shared memory into registers
            if (valid_primitive && valid_pixel && lane_idx == 0 && idx < config::block_size_blend) {
                const int current_shmem_index = i % 32;
                last_contributor = collected_last_contributor[current_shmem_index];
                const float4 color_pixel_after_transmittance = collected_color_pixel_after_transmittance[current_shmem_index];
                color_pixel_after = make_float3(color_pixel_after_transmittance);
                transmittance = color_pixel_after_transmittance.w;
                const float4 grad_info_pixel = collected_grad_info_pixel[current_shmem_index];
                grad_color_pixel = make_float3(grad_info_pixel);
                grad_alpha_common = grad_info_pixel.w;
            }

            const bool skip = !valid_primitive || !valid_pixel || idx < 0 || idx >= config::block_size_blend || tile_primitive_idx >= last_contributor;
            if (skip) continue;

            const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
            const float2 delta = mean2d - pixel;
            const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
            const float gaussian = expf(fminf(exponent, 0.0f));
            const float alpha = opacity * gaussian;
            if (alpha < config::min_alpha_threshold) continue;

            const float blending_weight = transmittance * alpha;

            // color gradient
            const float3 dL_dcolor = blending_weight * grad_color_pixel * color_grad_factor;
            dL_dcolor_accum += dL_dcolor;

            color_pixel_after -= blending_weight * color;

            // alpha gradient
            const float one_minus_alpha = 1.0f - alpha;
            const float one_minus_alpha_rcp = 1.0f / fmaxf(one_minus_alpha, config::one_minus_alpha_eps);
            const float dL_dalpha_from_color = dot(transmittance * color - color_pixel_after * one_minus_alpha_rcp, grad_color_pixel);
            const float dL_dalpha_from_alpha = grad_alpha_common * one_minus_alpha_rcp;
            const float dL_dalpha = dL_dalpha_from_color + dL_dalpha_from_alpha;
            // opacity gradient
            const float dL_dopacity = gaussian * dL_dalpha;
            dL_dopacity_accum += dL_dopacity;

            // conic and mean2d gradient
            const float gaussian_grad_helper = -alpha * dL_dalpha;
            const float3 dL_dconic = 0.5f * gaussian_grad_helper * make_float3(
                delta.x * delta.x,
                delta.x * delta.y,
                delta.y * delta.y
            );
            dL_dconic_accum += dL_dconic;
            const float2 dL_dmean2d = gaussian_grad_helper * make_float2(
                conic.x * delta.x + conic.y * delta.y,
                conic.y * delta.x + conic.z * delta.y
            );
            dL_dmean2d_accum += dL_dmean2d;

            transmittance *= one_minus_alpha;
        }

        // finally add the gradients using atomics
        if (valid_primitive) {
            atomicAdd(&grad_mean2d[primitive_idx].x, dL_dmean2d_accum.x);
            atomicAdd(&grad_mean2d[primitive_idx].y, dL_dmean2d_accum.y);
            atomicAdd(&grad_conic[primitive_idx], dL_dconic_accum.x);
            atomicAdd(&grad_conic[n_primitives + primitive_idx], dL_dconic_accum.y);
            atomicAdd(&grad_conic[2 * n_primitives + primitive_idx], dL_dconic_accum.z);
            const float dL_dopacity = opacity * (1.0f - opacity) * dL_dopacity_accum;
            atomicAdd(&grad_opacity[primitive_idx], dL_dopacity);
            atomicAdd(&grad_colors[primitive_idx].x, dL_dcolor_accum.x);
            atomicAdd(&grad_colors[primitive_idx].y, dL_dcolor_accum.y);
            atomicAdd(&grad_colors[primitive_idx].z, dL_dcolor_accum.z);
        }
    }

    template <uint n_attributes>
    __global__ void adam_step_invisible(
        const uint* __restrict__ primitive_n_touched_tiles,
        float* __restrict__ param,
        float2* __restrict__ moments,
        const int n_elements,
        const float step_size,
        const float bias_correction2_sqrt_rcp)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const uint primitive_idx = idx / n_attributes;
        if (idx >= n_elements || primitive_n_touched_tiles[primitive_idx] != 0) return;
        const float2 new_moments = moments[idx] * make_float2(config::beta1, config::beta2);
        const float denom = sqrtf(new_moments.y) * bias_correction2_sqrt_rcp + config::epsilon;
        param[idx] -= step_size * new_moments.x / denom;
        moments[idx] = new_moments;
    }

}
