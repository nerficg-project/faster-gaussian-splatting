import torch

from FasterGSCudaBackend import _C


def relocation_adjustment(
    old_opacities: torch.Tensor,
    old_scales: torch.Tensor,
    n_samples_per_primitive: torch.Tensor,
) -> 'tuple[torch.Tensor, torch.Tensor]':
    return _C.relocation_adjustment(old_opacities, old_scales, n_samples_per_primitive)


def add_noise(
    raw_scales: torch.Tensor,
    raw_rotations: torch.Tensor,
    raw_opacities: torch.Tensor,
    means: torch.Tensor,
    current_lr: float,
) -> None:
    _C.add_noise(raw_scales, raw_rotations, raw_opacities, means, current_lr)
