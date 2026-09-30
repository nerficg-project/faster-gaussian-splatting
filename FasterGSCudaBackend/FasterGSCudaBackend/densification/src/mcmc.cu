#include "mcmc.h"
#include "kernels_mcmc.cuh"
#include "densification_config.h"
#include "utils.h"
#include "helper_math.h"
#include <cstdint>

void faster_gs::densification::relocation_adjustment(
    const float* old_opacities,
    const float3* old_scales,
    const int64_t* n_samples_per_primitive,
    float* new_opacities,
    float3* new_scales,
    const int n_primitives)
{
    kernels::mcmc::init_relocation_coefficients();
    kernels::mcmc::relocation_cu<<<div_round_up(n_primitives, config::block_size_relocation), config::block_size_relocation>>>(
        old_opacities,
        old_scales,
        n_samples_per_primitive,
        new_opacities,
        new_scales,
        n_primitives
    );
    CHECK_CUDA(config::debug, "relocation_adjustment")
}

void faster_gs::densification::add_noise(
    const float3* raw_scales,
    const float4* raw_rotations,
    const float* raw_opacities,
    float3* means,
    const int n_primitives,
    const float current_lr,
    const uint64_t seed,
    const uint64_t offset)
{
    kernels::mcmc::add_noise_cu<<<div_round_up(n_primitives, config::block_size_add_noise), config::block_size_add_noise>>>(
        raw_scales,
        raw_rotations,
        raw_opacities,
        means,
        n_primitives,
        current_lr,
        seed,
        offset
    );
    CHECK_CUDA(config::debug, "add_noise")
}
