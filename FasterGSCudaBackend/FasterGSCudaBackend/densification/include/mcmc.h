#pragma once

#include "helper_math.h"
#include <cstdint>

namespace faster_gs::densification {

    void relocation_adjustment(
        const float* old_opacities,
        const float3* old_scales,
        const int64_t* n_samples_per_primitive,
        float* new_opacities,
        float3* new_scales,
        const int n_primitives);

    void add_noise(
        const float3* raw_scales,
        const float4* raw_rotations,
        const float* raw_opacities,
        float3* means,
        const int n_primitives,
        const float current_lr,
        const uint64_t seed,
        const uint64_t offset);

}
