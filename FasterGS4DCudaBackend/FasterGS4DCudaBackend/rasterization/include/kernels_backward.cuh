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
        const float3* __restrict__ spatial_means,
        const float* __restrict__ temporal_means,
        const float3* __restrict__ spatial_scales,
        const float* __restrict__ temporal_scales,
        const float4* __restrict__ left_isoclinic_rotations,
        const float4* __restrict__ right_isoclinic_rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_rest,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        const uint* __restrict__ primitive_n_touched_tiles,
        const float* __restrict__ primitive_cov3d,
        const float* __restrict__ primitive_mean3d,
        const float2* __restrict__ grad_mean2d,
        const float* __restrict__ grad_conic,
        float3* __restrict__ grad_spatial_means,
        float* __restrict__ grad_temporal_means,
        float3* __restrict__ grad_spatial_scales,
        float* __restrict__ grad_temporal_scales,
        float4* __restrict__ grad_left_isoclinic_rotations,
        float4* __restrict__ grad_right_isoclinic_rotations,
        float* __restrict__ grad_opacities,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
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
        const float timestamp)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;

        // load spatial 3d mean
        const float3 spatial_mean3d = spatial_means[primitive_idx]; // original paper uses spatial mean3d in the forward pass but the conditional mean3d here

        // sh evaluation backward
        const float3 dL_dmean3d_from_color = convert_sh_to_color_backward(
            sh_coefficients_rest, grad_sh_coefficients_0, grad_sh_coefficients_rest,
            spatial_mean3d, cam_position[0], primitive_idx, // TODO: this could be the conditional mean3d instead
            active_sh_bases, total_sh_bases
        );

        // load 3d mean
        const float3 mean3d = make_float3(
            primitive_mean3d[primitive_idx],
            primitive_mean3d[n_primitives + primitive_idx],
            primitive_mean3d[2 * n_primitives + primitive_idx]
        );

        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

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
        const mat3x3_triu cov3d = {
            primitive_cov3d[primitive_idx],
            primitive_cov3d[n_primitives + primitive_idx],
            primitive_cov3d[2 * n_primitives + primitive_idx],
            primitive_cov3d[3 * n_primitives + primitive_idx],
            primitive_cov3d[4 * n_primitives + primitive_idx],
            primitive_cov3d[5 * n_primitives + primitive_idx],
        };
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
        const float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1) + config::dilation,
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2) + config::dilation
        );
        const float aa = cov2d.x * cov2d.x, bb = cov2d.y * cov2d.y, cc = cov2d.z * cov2d.z;
        const float ac = cov2d.x * cov2d.z, ab = cov2d.x * cov2d.y, bc = cov2d.y * cov2d.z;
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

        // write total 3d mean gradient
        const float3 dL_dmean3d = dL_dmean3d_from_splatting + dL_dmean3d_from_color;
        grad_spatial_means[primitive_idx] = dL_dmean3d;

        // compute conditional 3d gaussian
        const float4 raw_left_isoclinic_rotation = left_isoclinic_rotations[primitive_idx];
        const float left_norm_sqrt_rcp = rsqrtf(dot(raw_left_isoclinic_rotation, raw_left_isoclinic_rotation));
        auto [a, b, c, d] = raw_left_isoclinic_rotation * left_norm_sqrt_rcp;
        const float4 raw_right_isoclinic_rotation = right_isoclinic_rotations[primitive_idx];
        const float right_norm_sqrt_rcp = rsqrtf(dot(raw_right_isoclinic_rotation, raw_right_isoclinic_rotation));
        auto [p, q, r, s] = raw_right_isoclinic_rotation * right_norm_sqrt_rcp;
        const mat4x4 R = {
            a*p - b*q - c*r - d*s, -a*q - b*p + c*s - d*r, -a*r - b*s - c*p + d*q, -a*s + b*r - c*q - d*p,
            a*q + b*p + c*s - d*r, a*p - b*q + c*r + d*s, a*s - b*r - c*q - d*p, -a*r - b*s + c*p - d*q,
            a*r - b*s + c*p + d*q, -a*s - b*r - c*q + d*p, a*p + b*q - c*r + d*s, a*q - b*p - c*s - d*r,
            a*s + b*r - c*q + d*p, a*r - b*s - c*p - d*q, -a*q + b*p - c*s - d*r, a*p + b*q + c*r - d*s
        };
        const float3 raw_spatial_scale = spatial_scales[primitive_idx];
        const float raw_temporal_scale = temporal_scales[primitive_idx];
        const float4 raw_scale = make_float4(raw_spatial_scale.x, raw_spatial_scale.y, raw_spatial_scale.z, raw_temporal_scale);
        const float4 variance = expf(2.0f * raw_scale);
        const mat4x4 RSS = {
            R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z, R.m14 * variance.w,
            R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z, R.m24 * variance.w,
            R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z, R.m34 * variance.w,
            R.m41 * variance.x, R.m42 * variance.y, R.m43 * variance.z, R.m44 * variance.w
        };
        // TODO: not all of these are used here
        const mat4x4_triu cov4d = {
            RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13 + RSS.m14 * R.m14,
            RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23 + RSS.m14 * R.m24,
            RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33 + RSS.m14 * R.m34,
            RSS.m11 * R.m41 + RSS.m12 * R.m42 + RSS.m13 * R.m43 + RSS.m14 * R.m44,
            RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23 + RSS.m24 * R.m24,
            RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33 + RSS.m24 * R.m34,
            RSS.m21 * R.m41 + RSS.m22 * R.m42 + RSS.m23 * R.m43 + RSS.m24 * R.m44,
            RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33 + RSS.m34 * R.m34,
            RSS.m31 * R.m41 + RSS.m32 * R.m42 + RSS.m33 * R.m43 + RSS.m34 * R.m44,
            RSS.m41 * R.m41 + RSS.m42 * R.m42 + RSS.m43 * R.m43 + RSS.m44 * R.m44
        };
        const float temporal_mean = temporal_means[primitive_idx];
        const float delta_t = timestamp - temporal_mean;
        const float conv4d_m44_rcp = 1.0f / cov4d.m44;
        const float marginal_t = expf(-0.5f * delta_t * delta_t * conv4d_m44_rcp);

        // opacity and marginal_t gradient
        const float raw_opacity = opacities[primitive_idx];
        const float opacity = sigmoid(raw_opacity);
        const float dL_dopacity_marginal_t = grad_opacities[primitive_idx];
        const float dL_dopacity = dL_dopacity_marginal_t * marginal_t * opacity * (1.0f - opacity);
        grad_opacities[primitive_idx] = dL_dopacity;
        const float dL_dmarginal_t = dL_dopacity_marginal_t * opacity;

        // temporal_mean and cov4d.m44 gradient from marginal_t
        const float dL_dtemporal_mean_from_marginal_t = dL_dmarginal_t * marginal_t * delta_t * conv4d_m44_rcp;
        const float dL_dcov4d_m44_from_marginal_t = dL_dmarginal_t * marginal_t * 0.5f * delta_t * delta_t * conv4d_m44_rcp * conv4d_m44_rcp;

        // temporal_mean and cov4d gradient from conditional 3d mean
        const float dL_dcov4d_m14_from_mean3d = dL_dmean3d_from_splatting.x * delta_t * conv4d_m44_rcp;
        const float dL_dcov4d_m24_from_mean3d = dL_dmean3d_from_splatting.y * delta_t * conv4d_m44_rcp;
        const float dL_dcov4d_m34_from_mean3d = dL_dmean3d_from_splatting.z * delta_t * conv4d_m44_rcp;
        const float dL_dtemporal_mean_from_mean3d = -conv4d_m44_rcp * dot(dL_dmean3d_from_splatting, make_float3(cov4d.m14, cov4d.m24, cov4d.m34));
        const float dL_dcov4d_m44_from_mean3d = delta_t * conv4d_m44_rcp * dL_dtemporal_mean_from_mean3d;

        // write total temporal mean gradient
        grad_temporal_means[primitive_idx] = dL_dtemporal_mean_from_marginal_t + dL_dtemporal_mean_from_mean3d;

        // cov4d gradient from conditional 3d covariance
        const float dL_dcov4d_m11_from_cov3d = dL_dcov3d.m11;
        const float dL_dcov4d_m12_from_cov3d = dL_dcov3d.m12;
        const float dL_dcov4d_m13_from_cov3d = dL_dcov3d.m13;
        const float dL_dcov4d_m14_from_cov3d = -conv4d_m44_rcp * (
            cov4d.m14 * dL_dcov3d.m11 + cov4d.m24 * dL_dcov3d.m12 + cov4d.m34 * dL_dcov3d.m13
        );
        const float dL_dcov4d_m22_from_cov3d = dL_dcov3d.m22;
        const float dL_dcov4d_m23_from_cov3d = dL_dcov3d.m23;
        const float dL_dcov4d_m24_from_cov3d = -conv4d_m44_rcp * (
            cov4d.m24 * dL_dcov3d.m22 + cov4d.m14 * dL_dcov3d.m12 + cov4d.m34 * dL_dcov3d.m23
        );
        const float dL_dcov4d_m33_from_cov3d = dL_dcov3d.m33;
        const float dL_dcov4d_m34_from_cov3d = -conv4d_m44_rcp * (
            cov4d.m34 * dL_dcov3d.m33 + cov4d.m14 * dL_dcov3d.m13 + cov4d.m24 * dL_dcov3d.m23
        );
        const float dL_dcov4d_m44_from_cov3d = conv4d_m44_rcp * conv4d_m44_rcp * (
            cov4d.m14 * cov4d.m14 * dL_dcov3d.m11 +
            2.0f * cov4d.m14 * cov4d.m24 * dL_dcov3d.m12 +
            2.0f * cov4d.m14 * cov4d.m34 * dL_dcov3d.m13 +
            cov4d.m24 * cov4d.m24 * dL_dcov3d.m22 +
            2.0f * cov4d.m24 * cov4d.m34 * dL_dcov3d.m23 +
            cov4d.m34 * cov4d.m34 * dL_dcov3d.m33
        );

        // symmetric version of full cov4d gradient
        const mat4x4_triu dL_dcov4d = {
            dL_dcov4d_m11_from_cov3d,
            dL_dcov4d_m12_from_cov3d,
            dL_dcov4d_m13_from_cov3d,
            dL_dcov4d_m14_from_cov3d + dL_dcov4d_m14_from_mean3d * 0.5f,
            dL_dcov4d_m22_from_cov3d,
            dL_dcov4d_m23_from_cov3d,
            dL_dcov4d_m24_from_cov3d + dL_dcov4d_m24_from_mean3d * 0.5f,
            dL_dcov4d_m33_from_cov3d,
            dL_dcov4d_m34_from_cov3d + dL_dcov4d_m34_from_mean3d * 0.5f,
            dL_dcov4d_m44_from_cov3d + dL_dcov4d_m44_from_mean3d + dL_dcov4d_m44_from_marginal_t
        };

        // scale gradient
        const float4 dL_dvariance = make_float4(
            R.m11 * R.m11 * dL_dcov4d.m11 + R.m21 * R.m21 * dL_dcov4d.m22 + R.m31 * R.m31 * dL_dcov4d.m33 + R.m41 * R.m41 * dL_dcov4d.m44 +
                2.0f * (
                    R.m11 * R.m21 * dL_dcov4d.m12 + R.m11 * R.m31 * dL_dcov4d.m13 + R.m11 * R.m41 * dL_dcov4d.m14 +
                    R.m21 * R.m31 * dL_dcov4d.m23 + R.m21 * R.m41 * dL_dcov4d.m24 +
                    R.m31 * R.m41 * dL_dcov4d.m34
                ),
            R.m12 * R.m12 * dL_dcov4d.m11 + R.m22 * R.m22 * dL_dcov4d.m22 + R.m32 * R.m32 * dL_dcov4d.m33 + R.m42 * R.m42 * dL_dcov4d.m44 +
                2.0f * (
                    R.m12 * R.m22 * dL_dcov4d.m12 + R.m12 * R.m32 * dL_dcov4d.m13 + R.m12 * R.m42 * dL_dcov4d.m14 +
                    R.m22 * R.m32 * dL_dcov4d.m23 + R.m22 * R.m42 * dL_dcov4d.m24 +
                    R.m32 * R.m42 * dL_dcov4d.m34
                ),
            R.m13 * R.m13 * dL_dcov4d.m11 + R.m23 * R.m23 * dL_dcov4d.m22 + R.m33 * R.m33 * dL_dcov4d.m33 + R.m43 * R.m43 * dL_dcov4d.m44 +
                2.0f * (
                    R.m13 * R.m23 * dL_dcov4d.m12 + R.m13 * R.m33 * dL_dcov4d.m13 + R.m13 * R.m43 * dL_dcov4d.m14 +
                    R.m23 * R.m33 * dL_dcov4d.m23 + R.m23 * R.m43 * dL_dcov4d.m24 +
                    R.m33 * R.m43 * dL_dcov4d.m34
                ),
            R.m14 * R.m14 * dL_dcov4d.m11 + R.m24 * R.m24 * dL_dcov4d.m22 + R.m34 * R.m34 * dL_dcov4d.m33 + R.m44 * R.m44 * dL_dcov4d.m44 +
                2.0f * (
                    R.m14 * R.m24 * dL_dcov4d.m12 + R.m14 * R.m34 * dL_dcov4d.m13 + R.m14 * R.m44 * dL_dcov4d.m14 +
                    R.m24 * R.m34 * dL_dcov4d.m23 + R.m24 * R.m44 * dL_dcov4d.m24 +
                    R.m34 * R.m44 * dL_dcov4d.m34
                )
        );
        const float4 dL_dscale = 2.0f * variance * dL_dvariance;
        grad_spatial_scales[primitive_idx] = make_float3(dL_dscale);
        grad_temporal_scales[primitive_idx] = dL_dscale.w;

        // rotation gradient
        const mat4x4 dL_dR = {
            2.0f * (RSS.m11 * dL_dcov3d.m11 + RSS.m21 * dL_dcov3d.m12 + RSS.m31 * dL_dcov3d.m13 + RSS.m41 * dL_dcov4d.m14),
            2.0f * (RSS.m12 * dL_dcov3d.m11 + RSS.m22 * dL_dcov3d.m12 + RSS.m32 * dL_dcov3d.m13 + RSS.m42 * dL_dcov4d.m14),
            2.0f * (RSS.m13 * dL_dcov3d.m11 + RSS.m23 * dL_dcov3d.m12 + RSS.m33 * dL_dcov3d.m13 + RSS.m43 * dL_dcov4d.m14),
            2.0f * (RSS.m14 * dL_dcov3d.m11 + RSS.m24 * dL_dcov3d.m12 + RSS.m34 * dL_dcov3d.m13 + RSS.m44 * dL_dcov4d.m14),

            2.0f * (RSS.m11 * dL_dcov3d.m12 + RSS.m21 * dL_dcov3d.m22 + RSS.m31 * dL_dcov3d.m23 + RSS.m41 * dL_dcov4d.m24),
            2.0f * (RSS.m12 * dL_dcov3d.m12 + RSS.m22 * dL_dcov3d.m22 + RSS.m32 * dL_dcov3d.m23 + RSS.m42 * dL_dcov4d.m24),
            2.0f * (RSS.m13 * dL_dcov3d.m12 + RSS.m23 * dL_dcov3d.m22 + RSS.m33 * dL_dcov3d.m23 + RSS.m43 * dL_dcov4d.m24),
            2.0f * (RSS.m14 * dL_dcov3d.m12 + RSS.m24 * dL_dcov3d.m22 + RSS.m34 * dL_dcov3d.m23 + RSS.m44 * dL_dcov4d.m24),

            2.0f * (RSS.m11 * dL_dcov3d.m13 + RSS.m21 * dL_dcov3d.m23 + RSS.m31 * dL_dcov3d.m33 + RSS.m41 * dL_dcov4d.m34),
            2.0f * (RSS.m12 * dL_dcov3d.m13 + RSS.m22 * dL_dcov3d.m23 + RSS.m32 * dL_dcov3d.m33 + RSS.m42 * dL_dcov4d.m34),
            2.0f * (RSS.m13 * dL_dcov3d.m13 + RSS.m23 * dL_dcov3d.m23 + RSS.m33 * dL_dcov3d.m33 + RSS.m43 * dL_dcov4d.m34),
            2.0f * (RSS.m14 * dL_dcov3d.m13 + RSS.m24 * dL_dcov3d.m23 + RSS.m34 * dL_dcov3d.m33 + RSS.m44 * dL_dcov4d.m34),

            2.0f * (RSS.m11 * dL_dcov4d.m14 + RSS.m21 * dL_dcov4d.m24 + RSS.m31 * dL_dcov4d.m34 + RSS.m41 * dL_dcov4d.m44),
            2.0f * (RSS.m12 * dL_dcov4d.m14 + RSS.m22 * dL_dcov4d.m24 + RSS.m32 * dL_dcov4d.m34 + RSS.m42 * dL_dcov4d.m44),
            2.0f * (RSS.m13 * dL_dcov4d.m14 + RSS.m23 * dL_dcov4d.m24 + RSS.m33 * dL_dcov4d.m34 + RSS.m43 * dL_dcov4d.m44),
            2.0f * (RSS.m14 * dL_dcov4d.m14 + RSS.m24 * dL_dcov4d.m24 + RSS.m34 * dL_dcov4d.m34 + RSS.m44 * dL_dcov4d.m44)
        };

        // left isoclinic rotation gradient
        const float dL_da = dL_dR.m11 * p + dL_dR.m12 * -q + dL_dR.m13 * -r + dL_dR.m14 * -s
                          + dL_dR.m21 * q + dL_dR.m22 * p + dL_dR.m23 * s + dL_dR.m24 * -r
                          + dL_dR.m31 * r + dL_dR.m32 * -s + dL_dR.m33 * p + dL_dR.m34 * q
                          + dL_dR.m41 * s + dL_dR.m42 * r + dL_dR.m43 * -q + dL_dR.m44 * p;
        const float dL_db = dL_dR.m11 * -q + dL_dR.m12 * -p + dL_dR.m13 * -s + dL_dR.m14 * r
                          + dL_dR.m21 * p + dL_dR.m22 * -q + dL_dR.m23 * -r + dL_dR.m24 * -s
                          + dL_dR.m31 * -s + dL_dR.m32 * -r + dL_dR.m33 * q + dL_dR.m34 * -p
                          + dL_dR.m41 * r + dL_dR.m42 * -s + dL_dR.m43 * p + dL_dR.m44 * q;
        const float dL_dc = dL_dR.m11 * -r + dL_dR.m12 * s + dL_dR.m13 * -p + dL_dR.m14 * -q
                          + dL_dR.m21 * s + dL_dR.m22 * r + dL_dR.m23 * -q + dL_dR.m24 * p
                          + dL_dR.m31 * p + dL_dR.m32 * -q + dL_dR.m33 * -r + dL_dR.m34 * -s
                          + dL_dR.m41 * -q + dL_dR.m42 * -p + dL_dR.m43 * -s + dL_dR.m44 * r;
        const float dL_dd = dL_dR.m11 * -s + dL_dR.m12 * -r + dL_dR.m13 * q + dL_dR.m14 * -p
                          + dL_dR.m21 * -r + dL_dR.m22 * s + dL_dR.m23 * -p + dL_dR.m24 * -q
                          + dL_dR.m31 * q + dL_dR.m32 * p + dL_dR.m33 * s + dL_dR.m34 * -r
                          + dL_dR.m41 * p + dL_dR.m42 * -q + dL_dR.m43 * -r + dL_dR.m44 * -s;
        const float4 dL_dleft_normalized = make_float4(dL_da, dL_db, dL_dc, dL_dd);
        const float4 left_normalized = make_float4(a, b, c, d);
        const float4 dL_dleft_isoclinic_rotation = (dL_dleft_normalized - dot(dL_dleft_normalized, left_normalized) * left_normalized) * left_norm_sqrt_rcp;
        grad_left_isoclinic_rotations[primitive_idx] = dL_dleft_isoclinic_rotation;

        // right isoclinic rotation gradient
        const float dL_dp = dL_dR.m11 * a + dL_dR.m12 * -b + dL_dR.m13 * -c + dL_dR.m14 * -d
                          + dL_dR.m21 * b + dL_dR.m22 * a + dL_dR.m23 * -d + dL_dR.m24 * c
                          + dL_dR.m31 * c + dL_dR.m32 * d + dL_dR.m33 * a + dL_dR.m34 * -b
                          + dL_dR.m41 * d + dL_dR.m42 * -c + dL_dR.m43 * b + dL_dR.m44 * a;
        const float dL_dq = dL_dR.m11 * -b + dL_dR.m12 * -a + dL_dR.m13 * d + dL_dR.m14 * -c
                          + dL_dR.m21 * a + dL_dR.m22 * -b + dL_dR.m23 * -c + dL_dR.m24 * -d
                          + dL_dR.m31 * d + dL_dR.m32 * -c + dL_dR.m33 * b + dL_dR.m34 * a
                          + dL_dR.m41 * -c + dL_dR.m42 * -d + dL_dR.m43 * -a + dL_dR.m44 * b;
        const float dL_dr = dL_dR.m11 * -c + dL_dR.m12 * -d + dL_dR.m13 * -a + dL_dR.m14 * b
                          + dL_dR.m21 * -d + dL_dR.m22 * c + dL_dR.m23 * -b + dL_dR.m24 * -a
                          + dL_dR.m31 * a + dL_dR.m32 * -b + dL_dR.m33 * -c + dL_dR.m34 * -d
                          + dL_dR.m41 * b + dL_dR.m42 * a + dL_dR.m43 * -d + dL_dR.m44 * c;
        const float dL_ds = dL_dR.m11 * -d + dL_dR.m12 * c + dL_dR.m13 * -b + dL_dR.m14 * -a
                          + dL_dR.m21 * c + dL_dR.m22 * d + dL_dR.m23 * a + dL_dR.m24 * -b
                          + dL_dR.m31 * -b + dL_dR.m32 * -a + dL_dR.m33 * d + dL_dR.m34 * -c
                          + dL_dR.m41 * a + dL_dR.m42 * -b + dL_dR.m43 * -c + dL_dR.m44 * -d;
        const float4 dL_dright_normalized = make_float4(dL_dp, dL_dq, dL_dr, dL_ds);
        const float4 right_normalized = make_float4(p, q, r, s);
        const float4 dL_dright_isoclinic_rotation = (dL_dright_normalized - dot(dL_dright_normalized, right_normalized) * right_normalized) * right_norm_sqrt_rcp;
        grad_right_isoclinic_rotations[primitive_idx] = dL_dright_isoclinic_rotation;

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
        float3* __restrict__ grad_sh_coefficients_0,
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
            atomicAdd(&grad_opacity[primitive_idx], dL_dopacity_accum);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].x, dL_dcolor_accum.x);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].y, dL_dcolor_accum.y);
            atomicAdd(&grad_sh_coefficients_0[primitive_idx].z, dL_dcolor_accum.z);
        }
    }

}
