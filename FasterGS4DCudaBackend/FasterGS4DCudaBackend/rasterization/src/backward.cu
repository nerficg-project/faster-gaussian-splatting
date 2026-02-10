#include "backward.h"
#include "kernels_backward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"

void faster_gs::rasterization::backward(
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
    const float timestamp)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    InstanceBuffers instance_buffers = InstanceBuffers::from_blob(instance_buffers_blob, n_instances, end_bit);
    BucketBuffers bucket_buffers = BucketBuffers::from_blob(bucket_buffers_blob, n_buckets);
    instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    kernels::backward::blend_backward_cu<<<n_buckets, 32>>>(
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
        n_primitives,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug, "blend_backward")

    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        spatial_means,
        temporal_means,
        spatial_scales,
        temporal_scales,
        left_isoclinic_rotations,
        right_isoclinic_rotations,
        opacities,
        sh_coefficients_rest,
        w2c,
        cam_position,
        primitive_buffers.n_touched_tiles,
        primitive_buffers.cov3d,
        primitive_buffers.mean3d,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_spatial_means,
        grad_temporal_means,
        grad_spatial_scales,
        grad_temporal_scales,
        grad_left_isoclinic_rotations,
        grad_right_isoclinic_rotations,
        grad_opacities,
        grad_sh_coefficients_0,
        grad_sh_coefficients_rest,
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
        timestamp
    );
    CHECK_CUDA(config::debug, "preprocess_backward")

}
