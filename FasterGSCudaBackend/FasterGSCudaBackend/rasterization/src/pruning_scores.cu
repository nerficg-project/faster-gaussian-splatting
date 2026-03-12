#include "pruning_scores.h"
#include "kernels_pruning_scores.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

// sorting is done separately for depth and tile as proposed in https://github.com/m-schuetz/Splatshop
void faster_gs::rasterization::pruning_scores(
    std::function<char* (size_t)> resize_primitive_buffers,
    std::function<char* (size_t)> resize_tile_buffers,
    std::function<char* (size_t)> resize_instance_buffers,
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_coefficients_0,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    float* scores,
    const int n_primitives,
    const int active_sh_bases,
    const int total_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    char* tile_buffers_blob = resize_tile_buffers(required<TileBuffers>(n_tiles));
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    }
    else cudaMemset(tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    char* primitive_buffers_blob = resize_primitive_buffers(required<PrimitiveBuffers>(n_primitives));
    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);

    cudaMemset(primitive_buffers.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(primitive_buffers.n_instances, 0, sizeof(uint));

    kernels::pruning_scores::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        primitive_buffers.depth_keys.Current(),
        primitive_buffers.primitive_indices.Current(),
        primitive_buffers.n_touched_tiles,
        primitive_buffers.screen_bounds,
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        primitive_buffers.color,
        primitive_buffers.n_visible_primitives,
        primitive_buffers.n_instances,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_sh_bases,
        static_cast<float>(width),
        static_cast<float>(height),
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing
    );
    CHECK_CUDA(config::debug, "preprocess")

    int n_visible_primitives;
    cudaMemcpy(&n_visible_primitives, primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    int n_instances;
    cudaMemcpy(&n_instances, primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    cub::DeviceRadixSort::SortPairs(
        primitive_buffers.cub_workspace,
        primitive_buffers.cub_workspace_size,
        primitive_buffers.depth_keys,
        primitive_buffers.primitive_indices,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (depth)")

    kernels::pruning_scores::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering>>>(
        primitive_buffers.primitive_indices.Current(),
        primitive_buffers.n_touched_tiles,
        primitive_buffers.offset,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug, "apply_depth_ordering")

    cub::DeviceScan::ExclusiveSum(
        primitive_buffers.cub_workspace,
        primitive_buffers.cub_workspace_size,
        primitive_buffers.offset,
        primitive_buffers.offset,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (primitive_buffers.offset)")

    // with 16x16 tiles, 16 bit keys are sufficient for up to 16M pixels, i.e., 4Kx4K images
    // beyond that, 32 bit keys are needed and for best performance, we template the remaining rasterization steps
    // note that with c++20 one could use a templated lambda to improve readability here
    #define COMPUTE_SCORES_ARGS \
        resize_instance_buffers, \
        primitive_buffers, \
        tile_buffers, \
        grid, \
        block, \
        bg_color, \
        scores, \
        memset_stream, \
        n_visible_primitives, \
        n_instances, \
        end_bit, \
        width, \
        height
    if (end_bit <= 16) compute_scores<ushort>(COMPUTE_SCORES_ARGS);
    else compute_scores<uint>(COMPUTE_SCORES_ARGS);
    #undef COMPUTE_SCORES_ARGS
}

template <typename KeyT>
void faster_gs::rasterization::compute_scores(
    std::function<char* (size_t)>& resize_instance_buffers,
    PrimitiveBuffers& primitive_buffers,
    TileBuffers& tile_buffers,
    const dim3& grid,
    const dim3& block,
    const float3* bg_color,
    float* scores,
    const cudaStream_t memset_stream,
    const int n_visible_primitives,
    const int n_instances,
    const int end_bit,
    const int width,
    const int height)
{
    char* instance_buffers_blob = resize_instance_buffers(required<InstanceBuffers<KeyT>>(n_instances, end_bit));
    InstanceBuffers<KeyT> instance_buffers = InstanceBuffers<KeyT>::from_blob(instance_buffers_blob, n_instances, end_bit);

    kernels::pruning_scores::create_instances_cu<KeyT><<<div_round_up(n_visible_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        primitive_buffers.primitive_indices.Current(),
        primitive_buffers.offset,
        primitive_buffers.screen_bounds,
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        instance_buffers.keys.Current(),
        instance_buffers.primitive_indices.Current(),
        grid.x,
        n_visible_primitives
    );
    CHECK_CUDA(config::debug, "create_instances")

    cub::DeviceRadixSort::SortPairs(
        instance_buffers.cub_workspace,
        instance_buffers.cub_workspace_size,
        instance_buffers.keys,
        instance_buffers.primitive_indices,
        n_instances,
        0, end_bit
    );
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (tile)")

    if constexpr (!config::debug) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::pruning_scores::extract_instance_ranges_cu<KeyT><<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            instance_buffers.keys.Current(),
            tile_buffers.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::pruning_scores::compute_scores_cu<<<grid, block>>>(
        tile_buffers.instance_ranges,
        instance_buffers.primitive_indices.Current(),
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        primitive_buffers.color,
        bg_color,
        scores,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug, "compute_scores")
}
