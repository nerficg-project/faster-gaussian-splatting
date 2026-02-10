#pragma once

#include "rasterization_config.h"
#include "kernel_utils.cuh"
#include "helper_math.h"

namespace faster_gs::rasterization::kernels {

    #define DEF inline constexpr float
    // degree 0
    DEF C0 = 0.28209479177387814f;
    // degree 1
    DEF C1 = 0.48860251190291987f;
    // degree 2
    DEF C2a = 1.0925484305920792f;
    DEF C2b = 0.94617469575755997f;
    DEF C2c = 0.31539156525251999f;
    DEF C2d = 0.54627421529603959f;
    DEF C2e = 1.8923493915151202f;
    // degree 3
    DEF C3a = 0.59004358992664352f;
    DEF C3b = 1.7701307697799304f;
    DEF C3c = 2.8906114426405538f;
    DEF C3d = 0.45704579946446572f;
    DEF C3e = 2.2852289973223288f;
    DEF C3f = 1.865881662950577f;
    DEF C3g = 1.1195289977703462f;
    DEF C3h = 1.4453057213202769f;
    DEF C3i = 3.5402615395598609f;
    DEF C3j = 4.5704579946446566f;
    DEF C3k = 5.597644988851731f;
    #undef DEF

    __device__ inline float3 convert_sh_to_color(
        const float3* __restrict__ sh_coefficients_0,
        const float3* __restrict__ sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint total_sh_bases_rest)
    {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L340
        float3 result = 0.5f + C0 * sh_coefficients_0[primitive_idx];
        if (active_sh_bases > 1) {
            const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_sh_bases_rest;
            auto [x, y, z] = normalize(position - cam_position);
            result = result - C1 * y * coefficients_ptr[0]
                            + C1 * z * coefficients_ptr[1]
                            - C1 * x * coefficients_ptr[2];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                result = result + C2a * xy * coefficients_ptr[3]
                                - C2a * yz * coefficients_ptr[4]
                                + (C2b * zz - C2c) * coefficients_ptr[5]
                                - C2a * xz * coefficients_ptr[6]
                                + C2d * (xx - yy) * coefficients_ptr[7];
                if (active_sh_bases > 9) {
                    result = result + y * (C3a * yy - C3b * xx) * coefficients_ptr[8]
                                    + C3c * xy * z * coefficients_ptr[9]
                                    + y * (C3d - C3e * zz) * coefficients_ptr[10]
                                    + z * (C3f * zz - C3g) * coefficients_ptr[11]
                                    + x * (C3d - C3e * zz) * coefficients_ptr[12]
                                    + C3h * z * (xx - yy) * coefficients_ptr[13]
                                    + x * (C3b * yy - C3a * xx) * coefficients_ptr[14];
                }
            }
        }
        return result;
    }

    __device__ inline float3 convert_sh_to_color_backward(
        float3* __restrict__ sh_coefficients_0,
        float3* __restrict__ sh_coefficients_rest,
        float2* __restrict__ moments_sh_coefficients_0,
        float2* __restrict__ moments_sh_coefficients_rest,
        const float3* __restrict__ grad_colors,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint total_sh_bases_rest,
        const float bias_correction1_rcp,
        const float bias_correction2_sqrt_rcp)
    {
        const float step_size_sh_coefficients_0 = config::lr_sh_coefficients_0 * bias_correction1_rcp;
        const float step_size_sh_coefficients_rest = config::lr_sh_coefficients_rest * bias_correction1_rcp;

        const float3* coefficients_ptr = sh_coefficients_rest + primitive_idx * total_sh_bases_rest;
        const float3 grad_color = grad_colors[primitive_idx];

        const float3 dL_dsh_coefficients_0 = C0 * grad_color;
        adam_step_helper<3, 0>(dL_dsh_coefficients_0.x, reinterpret_cast<float*>(sh_coefficients_0), moments_sh_coefficients_0, primitive_idx, step_size_sh_coefficients_0, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 1>(dL_dsh_coefficients_0.y, reinterpret_cast<float*>(sh_coefficients_0), moments_sh_coefficients_0, primitive_idx, step_size_sh_coefficients_0, bias_correction2_sqrt_rcp);
        adam_step_helper<3, 2>(dL_dsh_coefficients_0.z, reinterpret_cast<float*>(sh_coefficients_0), moments_sh_coefficients_0, primitive_idx, step_size_sh_coefficients_0, bias_correction2_sqrt_rcp);

        float3 dcolor_dposition = make_float3(0.0f);
        if (active_sh_bases > 1) {
            auto [x_raw, y_raw, z_raw] = position - cam_position;
            auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
            const float3 dL_dsh_coefficients_rest_0 = -C1 * y * grad_color;
            adam_step_helper<config::n_sh_bases_rest * 3, 0>(dL_dsh_coefficients_rest_0.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 1>(dL_dsh_coefficients_rest_0.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 2>(dL_dsh_coefficients_rest_0.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            const float3 dL_dsh_coefficients_rest_1 = C1 * z * grad_color;
            adam_step_helper<config::n_sh_bases_rest * 3, 3>(dL_dsh_coefficients_rest_1.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 4>(dL_dsh_coefficients_rest_1.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 5>(dL_dsh_coefficients_rest_1.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            const float3 dL_dsh_coefficients_rest_2 = -C1 * x * grad_color;
            adam_step_helper<config::n_sh_bases_rest * 3, 6>(dL_dsh_coefficients_rest_2.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 7>(dL_dsh_coefficients_rest_2.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            adam_step_helper<config::n_sh_bases_rest * 3, 8>(dL_dsh_coefficients_rest_2.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
            float3 grad_direction_x = -C1 * coefficients_ptr[2];
            float3 grad_direction_y = -C1 * coefficients_ptr[0];
            float3 grad_direction_z = C1 * coefficients_ptr[1];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                const float3 dL_dsh_coefficients_rest_3 = C2a * xy * grad_color;
                adam_step_helper<config::n_sh_bases_rest * 3, 9>(dL_dsh_coefficients_rest_3.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 10>(dL_dsh_coefficients_rest_3.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 11>(dL_dsh_coefficients_rest_3.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                const float3 dL_dsh_coefficients_rest_4 = -C2a * yz * grad_color;
                adam_step_helper<config::n_sh_bases_rest * 3, 12>(dL_dsh_coefficients_rest_4.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 13>(dL_dsh_coefficients_rest_4.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 14>(dL_dsh_coefficients_rest_4.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                const float3 dL_dsh_coefficients_rest_5 = (C2b * zz - C2c) * grad_color;
                adam_step_helper<config::n_sh_bases_rest * 3, 15>(dL_dsh_coefficients_rest_5.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 16>(dL_dsh_coefficients_rest_5.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 17>(dL_dsh_coefficients_rest_5.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                const float3 dL_dsh_coefficients_rest_6 = -C2a * xz * grad_color;
                adam_step_helper<config::n_sh_bases_rest * 3, 18>(dL_dsh_coefficients_rest_6.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 19>(dL_dsh_coefficients_rest_6.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 20>(dL_dsh_coefficients_rest_6.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                const float3 dL_dsh_coefficients_rest_7 = C2d * (xx - yy) * grad_color;
                adam_step_helper<config::n_sh_bases_rest * 3, 21>(dL_dsh_coefficients_rest_7.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 22>(dL_dsh_coefficients_rest_7.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                adam_step_helper<config::n_sh_bases_rest * 3, 23>(dL_dsh_coefficients_rest_7.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                grad_direction_x = grad_direction_x + C2a * y * coefficients_ptr[3]
                                                    - C2a * z * coefficients_ptr[6]
                                                    + C2a * x * coefficients_ptr[7];
                grad_direction_y = grad_direction_y + C2a * x * coefficients_ptr[3]
                                                    - C2a * z * coefficients_ptr[4]
                                                    - C2a * y * coefficients_ptr[7];
                grad_direction_z = grad_direction_z - C2a * y * coefficients_ptr[4]
                                                    + C2e * z * coefficients_ptr[5]
                                                    - C2a * x * coefficients_ptr[6];
                if (active_sh_bases > 9) {
                    const float3 dL_dsh_coefficients_rest_8 = y * (C3a * yy - C3b * xx) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 24>(dL_dsh_coefficients_rest_8.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 25>(dL_dsh_coefficients_rest_8.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 26>(dL_dsh_coefficients_rest_8.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_9 = C3c * xy * z * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 27>(dL_dsh_coefficients_rest_9.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 28>(dL_dsh_coefficients_rest_9.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 29>(dL_dsh_coefficients_rest_9.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_10 = y * (C3d - C3e * zz) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 30>(dL_dsh_coefficients_rest_10.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 31>(dL_dsh_coefficients_rest_10.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 32>(dL_dsh_coefficients_rest_10.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_11 = z * (C3f * zz - C3g) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 33>(dL_dsh_coefficients_rest_11.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 34>(dL_dsh_coefficients_rest_11.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 35>(dL_dsh_coefficients_rest_11.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_12 = x * (C3d - C3e * zz) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 36>(dL_dsh_coefficients_rest_12.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 37>(dL_dsh_coefficients_rest_12.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 38>(dL_dsh_coefficients_rest_12.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_13 = C3h * z * (xx - yy) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 39>(dL_dsh_coefficients_rest_13.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 40>(dL_dsh_coefficients_rest_13.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 41>(dL_dsh_coefficients_rest_13.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    const float3 dL_dsh_coefficients_rest_14 = x * (C3b * yy - C3a * xx) * grad_color;
                    adam_step_helper<config::n_sh_bases_rest * 3, 42>(dL_dsh_coefficients_rest_14.x, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 43>(dL_dsh_coefficients_rest_14.y, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    adam_step_helper<config::n_sh_bases_rest * 3, 44>(dL_dsh_coefficients_rest_14.z, reinterpret_cast<float*>(sh_coefficients_rest), moments_sh_coefficients_rest, primitive_idx, step_size_sh_coefficients_rest, bias_correction2_sqrt_rcp);
                    grad_direction_x = grad_direction_x - C3i * xy * coefficients_ptr[8]
                                                        + C3c * yz * coefficients_ptr[9]
                                                        + (C3d - C3e * zz) * coefficients_ptr[12]
                                                        + C3c * xz * coefficients_ptr[13]
                                                        + C3b * (yy - xx) * coefficients_ptr[14];
                    grad_direction_y = grad_direction_y + C3b * (yy - xx) * coefficients_ptr[8]
                                                        + C3c * xz * coefficients_ptr[9]
                                                        + (C3d - C3e * zz) * coefficients_ptr[10]
                                                        - C3c * yz * coefficients_ptr[13]
                                                        + C3i * xy * coefficients_ptr[14];
                    grad_direction_z = grad_direction_z + C3c * xy * coefficients_ptr[9]
                                                        - C3j * yz * coefficients_ptr[10]
                                                        + (C3k * zz - C3g) * coefficients_ptr[11]
                                                        - C3j * xz * coefficients_ptr[12]
                                                        + C3h * (xx - yy) * coefficients_ptr[13];
                }
            }

            const float3 grad_direction = make_float3(
                dot(grad_direction_x, grad_color),
                dot(grad_direction_y, grad_color),
                dot(grad_direction_z, grad_color)
            );
            const float xx_raw = x_raw * x_raw, yy_raw = y_raw * y_raw, zz_raw = z_raw * z_raw;
            const float xy_raw = x_raw * y_raw, xz_raw = x_raw * z_raw, yz_raw = y_raw * z_raw;
            const float norm_sq = xx_raw + yy_raw + zz_raw;
            dcolor_dposition = make_float3(
                (yy_raw + zz_raw) * grad_direction.x - xy_raw * grad_direction.y - xz_raw * grad_direction.z,
                -xy_raw * grad_direction.x + (xx_raw + zz_raw) * grad_direction.y - yz_raw * grad_direction.z,
                -xz_raw * grad_direction.x - yz_raw * grad_direction.y + (xx_raw + yy_raw) * grad_direction.z
            ) * rsqrtf(norm_sq * norm_sq * norm_sq);
        }
        return dcolor_dposition;
    }

}
