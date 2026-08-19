#pragma once

#include "helper_math.h"

#define DEF inline constexpr

namespace faster_gs::rasterization::config {
    DEF bool debug = false;
    // rendering constants
    DEF float dilation = 0.3f;
    DEF float dilation_proper_antialiasing = 0.1f;
    DEF bool detach_dilation_proper_antialiasing_from_cov2d = true; // note: detaching leads to more stable gradients and less degenerate Gaussians
    DEF float min_cov2d_determinant = 1e-6f; // note: backward pass includes factor of 1 / (determinant^2)
    DEF bool original_opacity_interpretation = true; // whether to interpret opacity as part of the Gaussian as in 3DGS or as a separate property
    DEF float one_minus_alpha_eps = 1e-6f;
    DEF float transmittance_threshold = 1e-4f;
    // choose truncation preset at compile time
    #define TRUNCATION_MODE 0 // 0: 3.33 sigma (original), 1: 1 sigma, 2: 2 sigma, 3: 3 sigma, 4: 4 sigma
    #if TRUNCATION_MODE == 0
    DEF float min_alpha_threshold_rcp = 255.0f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
    DEF float max_power_threshold = 5.54126354516f; // ln(min_alpha_threshold_rcp)
    #elif TRUNCATION_MODE == 1
    static_assert(!original_opacity_interpretation, "one sigma truncation not possible with original opacity interpretation");
    DEF float min_alpha_threshold_rcp = 2.71828182846f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.36787944117
    DEF float max_power_threshold = 1.0f; // ln(min_alpha_threshold_rcp)
    #elif TRUNCATION_MODE == 2
    static_assert(!original_opacity_interpretation, "two sigma truncation not possible with original opacity interpretation");
    DEF float min_alpha_threshold_rcp = 7.38905609893f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.13533528323
    DEF float max_power_threshold = 2.0f; // ln(min_alpha_threshold_rcp)
    #elif TRUNCATION_MODE == 3
    static_assert(!original_opacity_interpretation, "three sigma truncation not possible with original opacity interpretation");
    DEF float min_alpha_threshold_rcp = 90.0171313005f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.01110899653
    DEF float max_power_threshold = 4.5f; // ln(min_alpha_threshold_rcp)
    #elif TRUNCATION_MODE == 4
    DEF float min_alpha_threshold_rcp = 2980.95798704f;
    DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00033546262
    DEF float max_power_threshold = 8.0f; // ln(min_alpha_threshold_rcp)
    #else
    static_assert(false, "invalid truncation mode");
    #endif
    #undef TRUNCATION_MODE
    // block size constants
    DEF int block_size_preprocess = 128;
    DEF int block_size_preprocess_backward = 128;
    DEF int block_size_apply_depth_ordering = 256;
    DEF int block_size_create_instances = 256;
    DEF int block_size_extract_instance_ranges = 256;
    DEF int block_size_extract_bucket_counts = 256;
    DEF int tile_width = 16;
    DEF int tile_height = 16;
    DEF int block_size_blend = tile_width * tile_height;
    DEF int n_sequential_threshold = 4;
    // warp size constants
    DEF int warp_size = 32;
    DEF int warp_tile_width = 8;
    DEF int warp_tile_height = 4;
    DEF int subtile_per_row = tile_width / warp_tile_width;
    DEF int warp_fetch_size = warp_size * 6;
}

namespace config = faster_gs::rasterization::config;

#undef DEF
