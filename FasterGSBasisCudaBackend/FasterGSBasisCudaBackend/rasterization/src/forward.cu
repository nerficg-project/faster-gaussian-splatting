#include "forward.h"
#include "kernels_forward.cuh"
#include "buffer_utils.h"
#include "rasterization_config.h"
#include "utils.h"
#include "helper_math.h"
#include <cub/cub.cuh>
#include <functional>

std::tuple<int, int> faster_gs::rasterization::forward(
    std::function<char* (size_t)> resize_primitive_buffers,
    std::function<char* (size_t)> resize_tile_buffers,
    std::function<char* (size_t)> resize_instance_buffers,
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_coefficients,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    float* image,
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
    const float far_plane)
{
    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1) + 32;

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

    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        w2c,
        cam_position,
        primitive_buffers.n_touched_tiles,
        primitive_buffers.screen_bounds,
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        primitive_buffers.color,
        primitive_buffers.depth,
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
        far_plane
    );
    CHECK_CUDA(config::debug, "preprocess")

    cub::DeviceScan::InclusiveSum(
        primitive_buffers.cub_workspace,
        primitive_buffers.cub_workspace_size,
        primitive_buffers.n_touched_tiles,
        primitive_buffers.offset,
        n_primitives
    );
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum")

    int n_instances;
    cudaMemcpy(&n_instances, primitive_buffers.offset + n_primitives - 1, sizeof(int), cudaMemcpyDeviceToHost);

    char* instance_buffers_blob = resize_instance_buffers(required<InstanceBuffers>(n_instances, end_bit));
    InstanceBuffers instance_buffers = InstanceBuffers::from_blob(instance_buffers_blob, n_instances, end_bit);

    kernels::forward::create_instances_cu<<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
        primitive_buffers.n_touched_tiles,
        primitive_buffers.offset,
        primitive_buffers.screen_bounds,
        primitive_buffers.depth,
        instance_buffers.keys.Current(),
        instance_buffers.primitive_indices.Current(),
        grid.x,
        n_primitives
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
    CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs")

    if constexpr (!config::debug) cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            instance_buffers.keys.Current(),
            tile_buffers.instance_ranges,
            n_instances
        );
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::forward::blend_cu<<<grid, block>>>(
        tile_buffers.instance_ranges,
        instance_buffers.primitive_indices.Current(),
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        primitive_buffers.color,
        bg_color,
        image,
        tile_buffers.final_transmittances,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug, "blend")

    return {n_instances, instance_buffers.primitive_indices.selector};

}
