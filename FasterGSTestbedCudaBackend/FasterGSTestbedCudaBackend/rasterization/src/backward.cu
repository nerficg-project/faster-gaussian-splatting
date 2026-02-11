#include "backward.h"
#include "kernels_backward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <cstdint>

template <bool use_separate_sorting>
void faster_gs::rasterization::backward(
    const float* grad_image,
    const float* image,
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float3* sh_coefficients,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    char* bucket_buffers_blob,
    float3* grad_means,
    float3* grad_scales,
    float4* grad_rotations,
    float* grad_opacities,
    float3* grad_sh_coefficients,
    float3* grad_sh_coefficients_0,
    float3* grad_sh_coefficients_rest,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float* grad_color_helper,
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
    const bool use_fused_activations,
    const bool use_per_gaussian_backward)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = use_separate_sorting ? extract_end_bit(n_tiles - 1) : extract_end_bit(n_tiles - 1) + 32;

    using PrimitiveBuffers = std::conditional_t<use_separate_sorting, PrimitiveBuffersSplitSorting, PrimitiveBuffers>;
    using InstanceBuffers = std::conditional_t<use_separate_sorting, InstanceBuffers<ushort>, InstanceBuffers<uint64_t>>;
    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles, use_per_gaussian_backward);
    InstanceBuffers instance_buffers = InstanceBuffers::from_blob(instance_buffers_blob, n_instances, end_bit);
    BucketBuffers bucket_buffers = BucketBuffers::from_blob(bucket_buffers_blob, n_buckets);
    instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    if (use_per_gaussian_backward) {
        kernels::backward::per_gaussian_blend_backward_cu<<<n_buckets, 32>>>(
            tile_buffers.instance_ranges,
            tile_buffers.buckets_offset,
            instance_buffers.primitive_indices.Current(),
            primitive_buffers.mean2d,
            primitive_buffers.conic_opacity,
            primitive_buffers.color,
            bg_color,
            grad_image,
            image,
            tile_buffers.final_transmittances,
            tile_buffers.max_n_processed,
            tile_buffers.n_processed,
            bucket_buffers.tile_index,
            bucket_buffers.color_transmittance,
            grad_mean2d_helper,
            grad_conic_helper,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_color_helper,
            n_primitives,
            width,
            height,
            grid.x,
            use_fused_activations
        );
        CHECK_CUDA(config::debug, "per_gaussian_blend_backward")
    }
    else {
        kernels::backward::blend_backward_cu<<<grid, block>>>(
            tile_buffers.instance_ranges,
            instance_buffers.primitive_indices.Current(),
            primitive_buffers.mean2d,
            primitive_buffers.conic_opacity,
            primitive_buffers.color,
            bg_color,
            grad_image,
            image,
            tile_buffers.final_transmittances,
            grad_mean2d_helper,
            grad_conic_helper,
            grad_opacities,
            grad_sh_coefficients_0,
            grad_color_helper,
            n_primitives,
            width,
            height,
            grid.x,
            use_fused_activations
        );
        CHECK_CUDA(config::debug, "blend_backward")
    }

    const bool concatenated_sh = sh_coefficients != nullptr;
    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales,
        rotations,
        concatenated_sh ? sh_coefficients : sh_coefficients_rest,
        w2c,
        cam_position,
        primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_color_helper,
        grad_means,
        grad_scales,
        grad_rotations,
        grad_sh_coefficients_0,
        concatenated_sh ? grad_sh_coefficients : grad_sh_coefficients_rest,
        densification_info,
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        static_cast<float>(width),
        static_cast<float>(height),
        focal_x,
        focal_y,
        center_x,
        center_y,
        use_fused_activations
    );
    CHECK_CUDA(config::debug, "preprocess_backward")

}

template void faster_gs::rasterization::backward<true>(
    const float*,
    const float*,
    const float3*,
    const float3*,
    const float4*,
    const float3*,
    const float3*,
    const float4*,
    const float3*,
    const float3*,
    char*,
    char*,
    char*,
    char*,
    float3*,
    float3*,
    float4*,
    float*,
    float3*,
    float3*,
    float3*,
    float2*,
    float*,
    float*,
    float*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const float,
    const float,
    const float,
    const float,
    const bool,
    const bool);

template void faster_gs::rasterization::backward<false>(
    const float*,
    const float*,
    const float3*,
    const float3*,
    const float4*,
    const float3*,
    const float3*,
    const float4*,
    const float3*,
    const float3*,
    char*,
    char*,
    char*,
    char*,
    float3*,
    float3*,
    float4*,
    float*,
    float3*,
    float3*,
    float3*,
    float2*,
    float*,
    float*,
    float*,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const int,
    const float,
    const float,
    const float,
    const float,
    const bool,
    const bool);
