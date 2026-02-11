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

namespace faster_gs::rasterization::kernels::forward {

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0, // nullptr when using concatenated sh coefficients
        const float3* __restrict__ sh_coefficients,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        float* __restrict__ primitive_depth,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool use_tight_bounds,
        const bool use_opacity_based_bounds,
        const bool use_fused_activations)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives) return;

        primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // z culling
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) return;

        // load opacity
        const float raw_opacity = opacities[primitive_idx];
        const float opacity = use_fused_activations ? sigmoid(raw_opacity) : raw_opacity;
        if (opacity < config::min_alpha_threshold) return;

        // compute 3d covariance from scale and rotation
        const float3 raw_scale = scales[primitive_idx];
        const float3 variance = use_fused_activations ? expf(2.0f * raw_scale) : raw_scale * raw_scale;
        const float4 raw_rotation = rotations[primitive_idx];
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = use_fused_activations ? convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq) : convert_normalized_quaternion_to_rotation_matrix(raw_rotation);
        if (quaternion_norm_sq < 1e-8f) return;
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

        // compute 2d mean in normalized image coordinates
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // ewa splatting
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
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2)
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < config::min_cov2d_determinant) return; // or (determinant <= 0.0f) with explicit handling in backward
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant
        );

        // 2d mean in screen space
        const float2 mean2d = make_float2(
            x * focal_x + center_x,
            y * focal_y + center_y
        );

        // compute bounds
        uint4 screen_bounds;
        if (use_tight_bounds) {
            const float cutoff_factor = use_opacity_based_bounds ? 2.0f * logf(opacity * config::min_alpha_threshold_rcp) : config::max_cutoff_factor;
            const float extent_x = fmaxf(sqrtf(cov2d.x * cutoff_factor) - 0.5f, 0.0f);
            const float extent_y = fmaxf(sqrtf(cov2d.z * cutoff_factor) - 0.5f, 0.0f);
            screen_bounds = make_uint4(
                min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
                min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
                min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
                min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
            );
        }
        else {
            const float midpoint = 0.5f * (cov2d.x + cov2d.z);
            const float discriminant = midpoint * midpoint - determinant;
            const float max_eigenvalue = midpoint + sqrtf(fmaxf(0.1f, discriminant));
            const float radius = ceilf(3.0f * sqrtf(max_eigenvalue));
            screen_bounds = make_uint4(
                min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - radius) / static_cast<float>(config::tile_width))))), // x_min
                min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + radius) / static_cast<float>(config::tile_width))))), // x_max
                min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - radius) / static_cast<float>(config::tile_height))))), // y_min
                min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + radius) / static_cast<float>(config::tile_height))))) // y_max
            );
        }
        const uint n_touched_tiles = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles == 0) return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);
        float3 color;
        if (sh_coefficients_0 == nullptr) {
            // concatenated sh coefficients
            color = convert_sh_to_color(
                sh_coefficients,
                mean3d, cam_position[0],
                primitive_idx, active_sh_bases, total_sh_bases
            );
        }
        else {
            // separate sh coefficients
            color = convert_sh_to_color(
                sh_coefficients_0, sh_coefficients,
                mean3d, cam_position[0],
                primitive_idx, active_sh_bases, total_sh_bases
            );
        }
        primitive_color[primitive_idx] = color;
        primitive_depth[primitive_idx] = depth;
    }

    __global__ void preprocess_w_tile_based_culling_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0, // nullptr when using concatenated sh coefficients
        const float3* __restrict__ sh_coefficients,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        float* __restrict__ primitive_depth,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_sh_bases,
        const float width,
        const float height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool use_fused_activations)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();

        bool active = true;
        uint primitive_idx = thread_idx;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (active) primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

        // z culling
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_plane || depth > far_plane) active = false;

        // early exit if whole warp is inactive
        if (warp.ballot(active) == 0) return;

        // load opacity
        const float raw_opacity = opacities[primitive_idx];
        const float opacity = use_fused_activations ? sigmoid(raw_opacity) : raw_opacity;
        if (opacity < config::min_alpha_threshold) active = false;

        // compute 3d covariance from scale and rotation
        const float3 raw_scale = scales[primitive_idx];
        const float3 variance = use_fused_activations ? expf(2.0f * raw_scale) : raw_scale * raw_scale;
        const float4 raw_rotation = rotations[primitive_idx];
        float quaternion_norm_sq = 1.0f;
        const mat3x3 R = use_fused_activations ? convert_quaternion_to_rotation_matrix(raw_rotation, quaternion_norm_sq) : convert_normalized_quaternion_to_rotation_matrix(raw_rotation);
        if (quaternion_norm_sq < 1e-8f) active = false;
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

        // compute 2d mean in normalized image coordinates
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // ewa splatting
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
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2)
        );
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < config::min_cov2d_determinant) active = false; // or (determinant <= 0.0f) with explicit handling in backward
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant
        );

        // 2d mean in screen space
        const float2 mean2d = make_float2(
            x * focal_x + center_x,
            y * focal_y + center_y
        );

        // compute bounds
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);
        const float cutoff_factor = 2.0f * power_threshold;
        const float extent_x = fmaxf(sqrtf(cov2d.x * cutoff_factor) - 0.5f, 0.0f);
        const float extent_y = fmaxf(sqrtf(cov2d.z * cutoff_factor) - 0.5f, 0.0f);
        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))), // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))), // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height))))) // y_max
        );
        const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles_max == 0) active = false;

        // early exit if whole warp is inactive
        if (warp.ballot(active) == 0) return;

        // compute exact number of tiles the primitive overlaps
        const uint n_touched_tiles = compute_exact_n_touched_tiles(
            mean2d, conic, screen_bounds,
            power_threshold, n_touched_tiles_max, active
        );

         // cooperative threads no longer needed
        if (n_touched_tiles == 0 || !active) return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w)
        );
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);
        float3 color;
        if (sh_coefficients_0 == nullptr) {
            // concatenated sh coefficients
            color = convert_sh_to_color(
                sh_coefficients,
                mean3d, cam_position[0],
                primitive_idx, active_sh_bases, total_sh_bases
            );
        }
        else {
            // separate sh coefficients
            color = convert_sh_to_color(
                sh_coefficients_0, sh_coefficients,
                mean3d, cam_position[0],
                primitive_idx, active_sh_bases, total_sh_bases
            );
        }
        primitive_color[primitive_idx] = color;
        primitive_depth[primitive_idx] = depth;
    }

    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float* __restrict__ primitive_depths,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        const uint primitive_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (primitive_idx >= n_primitives || primitive_n_touched_tiles[primitive_idx] == 0) return;
        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];
        const uint depth_key = __float_as_uint(primitive_depths[primitive_idx]);
        for (uint y = screen_bounds.z; y < screen_bounds.w; y++) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; x++) {
                const uint tile_idx = y * grid_width + x;
                const uint64_t instance_key = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key);
                instance_keys[offset] = instance_key;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_w_load_balancing_cu(
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float* __restrict__ primitive_depths,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        uint primitive_idx = thread_idx;
        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (primitive_n_touched_tiles[primitive_idx] == 0) active = false;

        if (warp.ballot(active) == 0) return;

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;
        const uint depth_key = __float_as_uint(primitive_depths[primitive_idx]);

        uint current_write_offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            const uint tile_idx = tile_y * grid_width + tile_x;
            const uint64_t instance_key = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key);
            instance_keys[current_write_offset] = instance_key;
            instance_primitive_indices[current_write_offset] = primitive_idx;
            current_write_offset++;
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ uint collected_depth_keys[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;
        collected_depth_keys[thread_rank] = depth_key;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];
            const uint depth_key_coop = collected_depth_keys[read_offset_shared];

            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint instance_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);

            const uint remaining_instance_count = instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const bool write = instance_idx < instance_count_coop;
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask);
                    const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                    const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                    const uint tile_idx = tile_y * grid_width + tile_x;
                    const uint64_t instance_key = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key_coop);
                    instance_keys[write_offset] = instance_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                const uint n_written = __popc(write_ballot);
                current_write_offset_coop += n_written;
            }
            warp.sync();
        }
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_w_tile_based_culling_cu(
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float* __restrict__ primitive_depths,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        uint64_t* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_primitives)
    {
        constexpr uint warp_size = 32;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<warp_size>(block);
        const uint thread_idx = cg::this_grid().thread_rank();
        const uint thread_rank = block.thread_rank();
        const uint warp_idx = warp.meta_group_rank();
        const uint warp_start = warp_idx * warp_size;
        const uint lane_idx = warp.thread_rank();
        const uint previous_lanes_mask = (1 << lane_idx) - 1;

        uint primitive_idx = thread_idx;
        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (primitive_n_touched_tiles[primitive_idx] == 0) active = false;

        if (warp.ballot(active) == 0) return;

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;
        const uint depth_key = __float_as_uint(primitive_depths[primitive_idx]);
        const float2 mean2d = primitive_mean2d[primitive_idx];
        const float2 mean2d_shifted = mean2d - 0.5f;
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const float3 conic = make_float3(conic_opacity);
        const float opacity = conic_opacity.w;
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);

        uint current_write_offset = (primitive_idx == 0) ? 0 : primitive_offsets[primitive_idx - 1];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                const uint tile_idx = tile_y * grid_width + tile_x;
                const uint64_t instance_key = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key);
                instance_keys[current_write_offset] = instance_key;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ uint collected_depth_keys[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;
        collected_depth_keys[thread_rank] = depth_key;
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];
            const uint depth_key_coop = collected_depth_keys[read_offset_shared];
            const float2 mean2d_shifted_coop = collected_mean2d_shifted[read_offset_shared];
            const float4 conic_power_threshold_coop = collected_conic_power_threshold[read_offset_shared];

            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint instance_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);
            const float3 conic_coop = make_float3(conic_power_threshold_coop);
            const float power_threshold_coop = conic_power_threshold_coop.w;

            const uint remaining_instance_count = instance_count_coop - config::n_sequential_threshold;
            const uint n_iterations = div_round_up(remaining_instance_count, warp_size);
            for (uint i = 0; i < n_iterations; i++) {
                const uint instance_idx = i * warp_size + lane_idx + config::n_sequential_threshold;
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const bool write = instance_idx < instance_count_coop && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = warp.ballot(write);
                if (write) {
                    const uint write_offset = current_write_offset_coop + __popc(write_ballot & previous_lanes_mask);
                    const uint tile_idx = tile_y * grid_width + tile_x;
                    const uint64_t instance_key = (static_cast<uint64_t>(tile_idx) << 32) | static_cast<uint64_t>(depth_key_coop);
                    instance_keys[write_offset] = instance_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                const uint n_written = __popc(write_ballot);
                current_write_offset_coop += n_written;
            }
            warp.sync();
        }
    }

    __global__ void extract_instance_ranges_cu(
        const uint64_t* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const uint64_t instance_key = instance_keys[instance_idx];
        const uint instance_tile_idx = instance_key >> 32;
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint64_t previous_instance_key = instance_keys[instance_idx - 1];
            const uint previous_instance_tile_idx = previous_instance_key >> 32;
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void extract_bucket_counts(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_n_buckets,
        const uint n_tiles)
    {
        const uint tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (tile_idx >= n_tiles) return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
        tile_n_buckets[tile_idx] = n_buckets;
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ tile_final_transmittances,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        const uint2 tile_range = tile_instance_ranges[group_index.y * grid_width + group_index.x];
        for (int n_points_remaining = tile_range.y - tile_range.x, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_color[thread_rank] = color;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                const float alpha = opacity * gaussian;
                if (alpha < config::min_alpha_threshold) continue;

                // blend fragment into pixel color
                color_pixel += transmittance * alpha * collected_color[j];

                // update transmittance
                transmittance *= 1.0f - alpha;

                // early stopping
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
        if (inside) {
            // apply background color
            color_pixel += transmittance * bg_color[0];
            // store results
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint n_pixels = width * height;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            tile_final_transmittances[pixel_idx] = transmittance;
        }
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_and_write_bucket_data_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_buckets_offset,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float3* __restrict__ primitive_color,
        const float3* __restrict__ bg_color,
        float* __restrict__ image,
        float* __restrict__ tile_final_transmittances,
        uint* __restrict__ tile_max_n_processed,
        uint* __restrict__ tile_n_processed,
        uint* __restrict__ bucket_tile_index,
        float4* __restrict__ bucket_color_transmittance,
        const uint width,
        const uint height,
        const uint grid_width)
    {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;
        // setup tile info
        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;
        // setup bucket to tile mapping
        const int n_buckets = div_round_up(n_points_total, 32);
        uint bucket_offset = (tile_idx == 0) ? 0 : tile_buckets_offset[tile_idx - 1];
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets) bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }
        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_processed = 0;
        uint n_processed_and_used = 0;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend) break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_color[thread_rank] = color;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                // store current color and transmittance every 32 Gaussians
                if (j % 32 == 0) {
                    const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + thread_rank] = current_color_transmittance;
                    bucket_offset++;
                }

                // track the number of processed Gaussians
                n_processed++;

                // evaluate current Gaussian at pixel
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float opacity = conic_opacity.w;
                const float2 delta = collected_mean2d[j] - pixel;
                const float exponent = -0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) - conic.y * delta.x * delta.y;
                const float gaussian = expf(fminf(exponent, 0.0f));
                const float alpha = opacity * gaussian;
                if (alpha < config::min_alpha_threshold) continue;

                // blend fragment into pixel color
                color_pixel += transmittance * alpha * collected_color[j];

                // update transmittance
                transmittance *= 1.0f - alpha;

                // update the number of used Gaussians
                n_processed_and_used = n_processed;

                // early stopping
                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
        if (inside) {
            // apply background color
            color_pixel += transmittance * bg_color[0];
            // store results
            const uint pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const uint n_pixels = width * height;
            image[pixel_idx] = color_pixel.x;
            image[n_pixels + pixel_idx] = color_pixel.y;
            image[2 * n_pixels + pixel_idx] = color_pixel.z;
            tile_final_transmittances[pixel_idx] = transmittance;
            tile_n_processed[pixel_idx] = n_processed_and_used;
        }
        // max reduce the number of processed Gaussians per tile
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        n_processed_and_used = BlockReduce(temp_storage).Reduce(n_processed_and_used, cub::Max());
        if (thread_rank == 0) tile_max_n_processed[tile_idx] = n_processed_and_used;
    }

}
