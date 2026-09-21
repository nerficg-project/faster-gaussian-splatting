#include "filter3d.h"
#include "filter3d_config.h"
#include "helper_math.h"
#include "utils.h"


namespace faster_gs::filter3d {

    __global__ void update_3d_filter_cu(
        const float3* positions,
        const float4* w2c,
        float* filter_3d,
        bool* visibility_mask,
        const int n_points,
        const float left,
        const float right,
        const float top,
        const float bottom,
        const float near_plane,
        const float far_plane,
        const float distance2filter)
    {
        const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (point_idx >= n_points) return;
        const float3 position_world = positions[point_idx];
        const float4 w2c_r3 = w2c[2];
        const float z = dot(make_float3(w2c_r3), position_world) + w2c_r3.w;
        if (z < near_plane || z > far_plane) return;
        const float4 w2c_r1 = w2c[0];
        const float x_clip = dot(make_float3(w2c_r1), position_world) + w2c_r1.w;
        if (x_clip < left * z || x_clip > right * z) return;
        const float4 w2c_r2 = w2c[1];
        const float y_clip = dot(make_float3(w2c_r2), position_world) + w2c_r2.w;
        if (y_clip < top * z || y_clip > bottom * z) return;
        const float filter_3d_new = distance2filter * z;
        if (filter_3d[point_idx] < filter_3d_new) return;
        filter_3d[point_idx] = filter_3d_new;
        visibility_mask[point_idx] = true;
    }

    void update_3d_filter_wrapper(
        const torch::Tensor& positions,
        const torch::Tensor& w2c,
        torch::Tensor& filter_3d,
        torch::Tensor& visibility_mask,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const float clipping_tolerance,
        const float distance2filter)
    {
        const int n_points = positions.size(0);

        const float bounds_factor = clipping_tolerance + 0.5f;
        const float width_f = static_cast<float>(width);
        const float height_f = static_cast<float>(height);
        const float max_x_shifted = bounds_factor * width_f;
        const float max_y_shifted = bounds_factor * height_f;
        const float principal_offset_x = center_x - 0.5f * width_f;
        const float principal_offset_y = center_y - 0.5f * height_f;
        const float left = (-max_x_shifted - principal_offset_x) / focal_x;
        const float right = (max_x_shifted - principal_offset_x) / focal_x;
        const float top = (-max_y_shifted - principal_offset_y) / focal_y;
        const float bottom = (max_y_shifted - principal_offset_y) / focal_y;

        update_3d_filter_cu<<<div_round_up(n_points, config::block_size_update_3d_filter), config::block_size_update_3d_filter>>>(
            reinterpret_cast<const float3*>(positions.data_ptr<float>()),
            reinterpret_cast<const float4*>(w2c.data_ptr<float>()),
            filter_3d.data_ptr<float>(),
            visibility_mask.data_ptr<bool>(),
            n_points,
            left,
            right,
            top,
            bottom,
            near_plane,
            far_plane,
            distance2filter
        );
        CHECK_CUDA(config::debug, "update_3d_filter_cu");
    }

}
