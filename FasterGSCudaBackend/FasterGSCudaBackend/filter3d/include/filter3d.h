#pragma once

#include <torch/extension.h>

namespace faster_gs::filter3d {

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
        const float distance2filter);

}
