#pragma once

#include "densification_config.h"
#include "helper_math.h"
#include <curand_philox4x32_x.h>
#include <cstdint>

namespace faster_gs::densification::kernels::mcmc {

    // precompute coefficients for Eq. (9) of 3DGS-MCMC paper
    __constant__ float c_relocation_coefficients[config::mcmc_max_n_samples * config::mcmc_max_n_samples];
    bool relocation_coefficients_initialized = false;
    void init_relocation_coefficients() {
        if (relocation_coefficients_initialized) return;  // only copy once
        float relocation_coefficients[config::mcmc_max_n_samples * config::mcmc_max_n_samples] = {};
        for (int n = 0; n < config::mcmc_max_n_samples; n++) {
            double binom = 1.0f;
            double sign = 1.0f;
            for (int k = 0; k <= n; k++, sign = -sign) {
                const double coefficient = binom * sign * rsqrt(static_cast<double>(k + 1));
                relocation_coefficients[n * config::mcmc_max_n_samples + k] = static_cast<float>(coefficient);
                binom *= static_cast<double>(n - k) / static_cast<double>(k + 1);
            }
        }
        cudaMemcpyToSymbol(c_relocation_coefficients, relocation_coefficients, sizeof(relocation_coefficients));
        relocation_coefficients_initialized = true;
    }

    __global__ void relocation_cu(
        const float* __restrict__ old_opacities,
        const float3* __restrict__ old_scales,
        const int64_t* __restrict__ n_samples_per_primitive,
        float* __restrict__ new_opacities,
        float3* __restrict__ new_scales,
        const uint n_primitives)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        const float old_opacity = old_opacities[primitive_idx];
        const float3 old_scale = old_scales[primitive_idx];
        const int n_samples = clamp(static_cast<int>(n_samples_per_primitive[primitive_idx]), 1, config::mcmc_max_n_samples);

        // compute new opacity
        const float new_opacity = 1.0f - powf(1.0f - old_opacity, 1.0f / static_cast<float>(n_samples));
        new_opacities[primitive_idx] = new_opacity;
        // compute new scale
        float denominator = 0.0f;
        for (int n = 0; n < n_samples; n++) {
            float new_opacity_power = new_opacity;
            for (int k = 0; k <= n; k++, new_opacity_power *= new_opacity) {
                denominator += c_relocation_coefficients[n * config::mcmc_max_n_samples + k] * new_opacity_power;
            }
        }
        const float scaling_factor = old_opacity / denominator;
        const float3 new_scale = scaling_factor * old_scale;
        new_scales[primitive_idx] = new_scale;
    }

    struct mat3x3 {
        float m11, m12, m13;
        float m21, m22, m23;
        float m31, m32, m33;
    };

    struct __align__(8) mat3x3_triu {
        float m11, m12, m13, m22, m23, m33;
    };

    // box-muller transform of two uniformly distributed 32-bit integers into two standard normal samples
    __device__ inline float2 box_muller(const uint bits_u, const uint bits_v) {
        const float u = fmaf(static_cast<float>(bits_u), 2.3283064e-10f, 1.1641532e-10f); // (0, 1]
        const float radius = sqrtf(-2.0f * logf(u));
        float s, c;
        sincosf(static_cast<float>(bits_v) * 1.4629181e-9f, &s, &c); // [0, 2pi]
        return make_float2(radius * c, radius * s);
    }

    __global__ void add_noise_cu(
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float* __restrict__ raw_opacities,
        float3* __restrict__ means,
        const uint n_primitives,
        const float current_lr,
        const uint64_t seed,
        const uint64_t offset)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        // compute 3d covariance from scale and rotation
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 variance = expf(2.0f * raw_scale);
        const float4 raw_rotation = raw_rotations[primitive_idx];
        auto [r, x, y, z] = raw_rotation;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float rx = r * x, ry = r * y, rz = r * z;
        const float norm_sq = r * r + xx + yy + zz;
        if (norm_sq < 1e-8f) return;
        const float norm_sq_rcp = 1.0f / norm_sq;
        const mat3x3 R = {
            1.0f - 2.0f * (yy + zz) * norm_sq_rcp, 2.0f * (xy - rz) * norm_sq_rcp, 2.0f * (xz + ry) * norm_sq_rcp,
            2.0f * (xy + rz) * norm_sq_rcp, 1.0f - 2.0f * (xx + zz) * norm_sq_rcp, 2.0f * (yz - rx) * norm_sq_rcp,
            2.0f * (xz - ry) * norm_sq_rcp, 2.0f * (yz + rx) * norm_sq_rcp, 1.0f - 2.0f * (xx + yy) * norm_sq_rcp
        };
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

        // sample noise using one philox evaluation per primitive
        const uint64_t counter = offset / 4;
        const uint4 bits = curand_Philox4x32_10(
            make_uint4(static_cast<uint>(counter), static_cast<uint>(counter >> 32), primitive_idx, 0u),
            make_uint2(static_cast<uint>(seed), static_cast<uint>(seed >> 32))
        );
        const float2 noise_xy = box_muller(bits.x, bits.y);
        const float3 noise = make_float3(noise_xy.x, noise_xy.y, box_muller(bits.z, bits.w).x);

        // transform noise
        const float3 transformed_noise = make_float3(
            dot(make_float3(cov3d.m11, cov3d.m12, cov3d.m13), noise),
            dot(make_float3(cov3d.m12, cov3d.m22, cov3d.m23), noise),
            dot(make_float3(cov3d.m13, cov3d.m23, cov3d.m33), noise)
        );

        // compute opacity-based noise scaling factor
        const float raw_opacity = raw_opacities[primitive_idx];
        const float opacity = 1.0f / (1.0f + expf(-raw_opacity));
        const float op_sigmoid = 1.0f / (1.0f + expf(100.0f * opacity - 0.5f));
        const float noise_factor = current_lr * op_sigmoid;

        // add noise to mean
        means[primitive_idx] += noise_factor * transformed_noise;
    }

}
