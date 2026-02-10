#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace faster_gs::rasterization::config {
    DEF bool debug = false;
    // rendering constants
    DEF float dilation = 0.3f;
    DEF float min_cov2d_determinant = 1e-6f; // note: backward pass includes factor of 1 / (determinant^2)
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float one_minus_alpha_eps = 1e-6f;
    DEF float transmittance_threshold = 1e-4f;
    // block size constants
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_apply_depth_ordering = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_extract_bucket_counts = 256;
    DEF int block_size_adam_step_invisible = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    DEF int n_sequential_threshold = 4;
    // optimization constants
    DEF float beta1 = 0.9f;
    DEF float beta2 = 0.999f;
    DEF float epsilon = 1e-15f;
    DEF float lr_sh_coefficients_0 = 0.0025f;
    DEF float lr_sh_coefficients_rest = 0.000125f; // 0.0025 / 20;
    DEF float lr_opacities = 0.025f; // recently updated in official code; used to be 0.05
    DEF float lr_scales = 0.005f;
    DEF float lr_rotations = 0.001f;
    DEF int n_sh_bases_rest = 15; // change based on used sh degree D: D=0 -> 0, D=1 -> 3, D=2 -> 8, D=3 -> 15
}

namespace config = faster_gs::rasterization::config;

#undef DEF
