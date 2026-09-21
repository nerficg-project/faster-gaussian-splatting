import torch

from FasterGSCudaBackend import _C


def update_3d_filter(
    positions: torch.Tensor,
    w2c: torch.Tensor,
    filter_3d: torch.Tensor,
    visibility_mask: torch.Tensor,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    center_x: float,
    center_y: float,
    near_plane: float,
    far_plane: float,
    clipping_tolerance: float,
    distance2filter: float,
) -> None:
    _C.update_3d_filter(
        positions,
        w2c,
        filter_3d,
        visibility_mask,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        clipping_tolerance,
        distance2filter,
    )
