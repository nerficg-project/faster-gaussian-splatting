#pragma once

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
        const float3* __restrict__ sh_coefficients_rest,
        float3* __restrict__ grad_sh_coefficients_0,
        float3* __restrict__ grad_sh_coefficients_rest,
        const float3& position,
        const float3& cam_position,
        const uint primitive_idx,
        const uint active_sh_bases,
        const uint total_sh_bases_rest)
    {
        // computation adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/212104156403bd87616c1a4f73a1c5f2c2e172a9/include/tiny-cuda-nn/common_device.h#L421
        const uint coefficients_base_idx = primitive_idx * total_sh_bases_rest;
        const float3* coefficients_ptr = sh_coefficients_rest + coefficients_base_idx;
        float3* grad_coefficients_ptr = grad_sh_coefficients_rest + coefficients_base_idx;
        const float3 grad_color = grad_sh_coefficients_0[primitive_idx];
        grad_sh_coefficients_0[primitive_idx] = C0 * grad_color;
        float3 dcolor_dposition = make_float3(0.0f);
        if (active_sh_bases > 1) {
            auto [x_raw, y_raw, z_raw] = position - cam_position;
            auto [x, y, z] = normalize(make_float3(x_raw, y_raw, z_raw));
            grad_coefficients_ptr[0] = -C1 * y * grad_color;
            grad_coefficients_ptr[1] = C1 * z * grad_color;
            grad_coefficients_ptr[2] = -C1 * x * grad_color;
            float3 grad_direction_x = -C1 * coefficients_ptr[2];
            float3 grad_direction_y = -C1 * coefficients_ptr[0];
            float3 grad_direction_z = C1 * coefficients_ptr[1];
            if (active_sh_bases > 4) {
                const float xx = x * x, yy = y * y, zz = z * z;
                const float xy = x * y, xz = x * z, yz = y * z;
                grad_coefficients_ptr[3] = C2a * xy * grad_color;
                grad_coefficients_ptr[4] = -C2a * yz * grad_color;
                grad_coefficients_ptr[5] = (C2b * zz - C2c) * grad_color;
                grad_coefficients_ptr[6] = -C2a * xz * grad_color;
                grad_coefficients_ptr[7] = C2d * (xx - yy) * grad_color;
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
                    grad_coefficients_ptr[8] = y * (C3a * yy - C3b * xx) * grad_color;
                    grad_coefficients_ptr[9] = C3c * xy * z * grad_color;
                    grad_coefficients_ptr[10] = y * (C3d - C3e * zz) * grad_color;
                    grad_coefficients_ptr[11] = z * (C3f * zz - C3g) * grad_color;
                    grad_coefficients_ptr[12] = x * (C3d - C3e * zz) * grad_color;
                    grad_coefficients_ptr[13] = C3h * z * (xx - yy) * grad_color;
                    grad_coefficients_ptr[14] = x * (C3b * yy - C3a * xx) * grad_color;
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
