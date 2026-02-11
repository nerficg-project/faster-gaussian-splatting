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

    __global__ void preprocess_separate_sorting_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0, // nullptr when using concatenated sh coefficients
        const float3* __restrict__ sh_coefficients,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        uint* __restrict__ n_visible_primitives,
        uint* __restrict__ n_instances,
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

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void preprocess_w_tile_based_culling_separate_sorting_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float* __restrict__ opacities,
        const float3* __restrict__ sh_coefficients_0, // nullptr when using concatenated sh coefficients
        const float3* __restrict__ sh_coefficients,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float3* __restrict__ primitive_color,
        uint* __restrict__ n_visible_primitives,
        uint* __restrict__ n_instances,
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

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void apply_depth_ordering_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_n_touched_tiles,
        uint* __restrict__ primitive_offset,
        const uint n_visible_primitives)
    {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    template <typename KeyT>
    __global__ void create_instances_separate_sorting_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives)
    {
        const uint thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_idx >= n_visible_primitives) return;
        const uint primitive_idx = primitive_indices_sorted[thread_idx];
        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        uint offset = primitive_offsets[thread_idx];
        for (uint y = screen_bounds.z; y < screen_bounds.w; y++) {
            for (uint x = screen_bounds.x; x < screen_bounds.y; x++) {
                const uint tile_idx = y * grid_width + x;
                const KeyT instance_key = static_cast<KeyT>(tile_idx);
                instance_keys[offset] = instance_key;
                instance_primitive_indices[offset] = primitive_idx;
                offset++;
            }
        }
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    template <typename KeyT>
    __global__ void create_instances_w_load_balancing_separate_sorting_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives)
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

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible_primitives) {
            active = false;
            original_idx = n_visible_primitives - 1;
        }

        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;

        uint current_write_offset = primitive_offsets[original_idx];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            const uint tile_idx = tile_y * grid_width + tile_x;
            const KeyT instance_key = static_cast<KeyT>(tile_idx);
            instance_keys[current_write_offset] = instance_key;
            instance_primitive_indices[current_write_offset] = primitive_idx;
            current_write_offset++;
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];

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
                    const KeyT instance_key = static_cast<KeyT>(tile_idx);
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
    template <typename KeyT>
    __global__ void create_instances_w_tile_based_culling_separate_sorting_cu(
        const uint* __restrict__ primitive_indices_sorted,
        const uint* __restrict__ primitive_offsets,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        KeyT* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives)
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

        uint original_idx = thread_idx;
        bool active = true;
        if (original_idx >= n_visible_primitives) {
            active = false;
            original_idx = n_visible_primitives - 1;
        }

        if (warp.ballot(active) == 0) return;

        const uint primitive_idx = primitive_indices_sorted[original_idx];

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint instance_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;
        const float2 mean2d = primitive_mean2d[primitive_idx];
        const float2 mean2d_shifted = mean2d - 0.5f;
        const float4 conic_opacity = primitive_conic_opacity[primitive_idx];
        const float3 conic = make_float3(conic_opacity);
        const float opacity = conic_opacity.w;
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);

        uint current_write_offset = primitive_offsets[original_idx];

        for (uint instance_idx = 0; active && instance_idx < instance_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
            const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
            const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
            if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                const uint tile_idx = tile_y * grid_width + tile_x;
                const KeyT instance_key = static_cast<KeyT>(tile_idx);
                instance_keys[current_write_offset] = instance_key;
                instance_primitive_indices[current_write_offset] = primitive_idx;
                current_write_offset++;
            }
        }

        const bool compute_cooperatively = active && instance_count > config::n_sequential_threshold;
        const uint remaining_threads = warp.ballot(compute_cooperatively);
        if (remaining_threads == 0) return;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_power_threshold[config::block_size_create_instances];
        collected_screen_bounds[thread_rank] = screen_bounds;
        collected_mean2d_shifted[thread_rank] = mean2d_shifted;
        collected_conic_power_threshold[thread_rank] = make_float4(conic, power_threshold);

        const uint n_remaining_threads = __popc(remaining_threads);
        for (uint n = 0; n < n_remaining_threads && n < warp_size; n++) {
            const uint current_lane = __fns(remaining_threads, 0, n + 1);
            const uint primitive_idx_coop = warp.shfl(primitive_idx, current_lane);
            uint current_write_offset_coop = warp.shfl(current_write_offset, current_lane);

            const uint read_offset_shared = warp_start + current_lane;
            const ushort4 screen_bounds_coop = collected_screen_bounds[read_offset_shared];
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
                    const KeyT instance_key = static_cast<KeyT>(tile_idx);
                    instance_keys[write_offset] = instance_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                const uint n_written = __popc(write_ballot);
                current_write_offset_coop += n_written;
            }
            warp.sync();
        }
    }

    template <typename KeyT>
    __global__ void extract_instance_ranges_separate_sorting_cu(
        const KeyT* __restrict__ instance_keys,
        uint2* __restrict__ tile_instance_ranges,
        const uint n_instances)
    {
        const uint instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (instance_idx >= n_instances) return;
        const KeyT instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const KeyT previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

}
