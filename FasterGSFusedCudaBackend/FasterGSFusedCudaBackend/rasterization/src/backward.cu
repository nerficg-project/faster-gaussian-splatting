#include "backward.h"
#include "kernels_backward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <stdexcept>

void faster_gs::rasterization::backward(
    const float* grad_image,
    const float* image,
    float3* means,
    float3* scales,
    float4* rotations,
    float* opacities,
    float3* sh_coefficients_0,
    float3* sh_coefficients_rest,
    float2* moments_means,
    float2* moments_scales,
    float2* moments_rotations,
    float2* moments_opacities,
    float2* moments_sh_coefficients_0,
    float2* moments_sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    char* bucket_buffers_blob,
    float* grad_opacities,
    float3* grad_colors,
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
    const float current_mean_lr,
    const int adam_step_count)
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
        grad_colors,
        n_primitives,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug, "blend_backward")

    const float bias_correction1_rcp = 1.0f / (1.0f - std::pow(config::beta1, adam_step_count));
    const float bias_correction2_sqrt_rcp = 1.0f / std::sqrt(1.0f - std::pow(config::beta2, adam_step_count));

    const float step_size_means = current_mean_lr * bias_correction1_rcp;

    kernels::backward::preprocess_backward_cu<<<div_round_up(n_primitives, config::block_size_preprocess_backward), config::block_size_preprocess_backward>>>(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        moments_means,
        moments_scales,
        moments_rotations,
        moments_opacities,
        moments_sh_coefficients_0,
        moments_sh_coefficients_rest,
        w2c,
        cam_position,
        primitive_buffers.n_touched_tiles,
        grad_mean2d_helper,
        grad_conic_helper,
        grad_opacities,
        grad_colors,
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
        step_size_means,
        bias_correction1_rcp,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "preprocess_backward")

    const int n_elements_means = n_primitives * 3;
    kernels::backward::adam_step_invisible<3><<<div_round_up(n_elements_means, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        reinterpret_cast<float*>(means),
        moments_means,
        n_elements_means,
        step_size_means,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (means)")

    const float step_size_scales = config::lr_scales * bias_correction1_rcp;
    const int n_elements_scales = n_primitives * 3;
    kernels::backward::adam_step_invisible<3><<<div_round_up(n_elements_scales, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        reinterpret_cast<float*>(scales),
        moments_scales,
        n_elements_scales,
        step_size_scales,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (scales)")

    const float step_size_rotations = config::lr_rotations * bias_correction1_rcp;
    const int n_elements_rotations = n_primitives * 4;
    kernels::backward::adam_step_invisible<4><<<div_round_up(n_elements_rotations, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        reinterpret_cast<float*>(rotations),
        moments_rotations,
        n_elements_rotations,
        step_size_rotations,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (rotations)")

    const float step_size_opacities = config::lr_opacities * bias_correction1_rcp;
    const int n_elements_opacities = n_primitives;
    kernels::backward::adam_step_invisible<1><<<div_round_up(n_elements_opacities, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        opacities,
        moments_opacities,
        n_elements_opacities,
        step_size_opacities,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (opacities)")

    const float step_size_sh_coefficients_0 = config::lr_sh_coefficients_0 * bias_correction1_rcp;
    const int n_elements_sh_coefficients_0 = n_primitives * 3;
    kernels::backward::adam_step_invisible<3><<<div_round_up(n_elements_sh_coefficients_0, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        reinterpret_cast<float*>(sh_coefficients_0),
        moments_sh_coefficients_0,
        n_elements_sh_coefficients_0,
        step_size_sh_coefficients_0,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (sh_coefficients_0)");

    if (active_sh_bases <= 1) return;

    if (total_sh_bases != config::n_sh_bases_rest)
        throw std::runtime_error(
            "The number of SH bases does not match \"n_sh_bases_rest\" constant. Please modify \"rasterization_config.h\" and recompile the rasterizer."
        );

    const float step_size_sh_coefficients_rest = config::lr_sh_coefficients_rest * bias_correction1_rcp;
    constexpr int elements_per_primitive_sh_coefficients_rest = config::n_sh_bases_rest * 3; // TODO: make this dynamic
    const int n_elements_sh_coefficients_rest = n_primitives * elements_per_primitive_sh_coefficients_rest;
    kernels::backward::adam_step_invisible<elements_per_primitive_sh_coefficients_rest><<<div_round_up(n_elements_sh_coefficients_rest, config::block_size_adam_step_invisible), config::block_size_adam_step_invisible>>>(
        primitive_buffers.n_touched_tiles,
        reinterpret_cast<float*>(sh_coefficients_rest),
        moments_sh_coefficients_rest,
        n_elements_sh_coefficients_rest,
        step_size_sh_coefficients_rest,
        bias_correction2_sqrt_rcp
    );
    CHECK_CUDA(config::debug, "adam_step_invisible (sh_coefficients_rest)")

}
