"""FasterGS4D/utils.py"""

import io
import warnings
import contextlib

import torch

from Datasets.Base import BaseDataset
from Logging import Logger


def enable_expandable_segments() -> bool:
    """Return True if 'expandable_segments' allocator feature is available on this device."""
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer), warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        torch.empty(1, device='cuda')  # allocate gpu memory to trigger potential warning
    stderr_output = stderr_buffer.getvalue()
    for warning in caught_warnings:
        if 'expandable_segments' in str(warning.message):
            return False
    if 'expandable_segments' in stderr_output:
        return False
    return True


def carve(points: torch.Tensor, dataset: BaseDataset, in_all_frustums: bool, enforce_alpha: bool) -> torch.Tensor:
    """
    Carves away points based on visibility and alpha.
    - Points that are never in-frustum in any view are removed.
    - If in_all_frustums=True, points not in-frustum in all views are removed.
    - If enforce_alpha=True, points that project to a pixel with alpha=0 in any view (where the point is in-frustum) are removed.
    """
    Logger.log_info(f'removing points that would not be visible in any training view (in_all_frustums={in_all_frustums}, enforce_alpha={enforce_alpha})')
    in_frustum_any = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    in_frustum_all = torch.ones_like(in_frustum_any)
    in_alpha_all = torch.ones_like(in_frustum_any)
    dilation_kernel = torch.ones(1, 1, 3, 3) if enforce_alpha else None
    for view in dataset:
        xy_screen, _, in_frustum = view.project_points(points)
        in_frustum_any |= in_frustum
        if in_all_frustums:
            in_frustum_all &= in_frustum
        if enforce_alpha and in_frustum.any() and (alpha_gt := view.alpha) is not None:
            alpha_gt = torch.nn.functional.conv2d(alpha_gt[None], dilation_kernel, padding=1)[0] > 0
            xy_screen = torch.floor(xy_screen[in_frustum]).long()
            valid_alpha = alpha_gt[0, xy_screen[:, 1], xy_screen[:, 0]] > 0
            in_alpha_all[in_frustum] &= valid_alpha
    valid_mask = in_frustum_any & in_alpha_all & in_frustum_all
    return points[valid_mask].contiguous()


def quaternion_to_left_isoclinic(quaternions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # add batch dimension if not present
    if batch_dim_added := quaternions.dim() == 1:
        quaternions = quaternions[None]
    if normalize:
        quaternions = torch.nn.functional.normalize(quaternions)
    R = torch.empty((quaternions.shape[0], 4, 4), dtype=quaternions.dtype, device=quaternions.device)
    a, b, c, d = torch.unbind(quaternions, dim=1)

    R[:, 0, 0] = a
    R[:, 0, 1] = -b
    R[:, 0, 2] = -c
    R[:, 0, 3] = -d
    R[:, 1, 0] = b
    R[:, 1, 1] = a
    R[:, 1, 2] = -d
    R[:, 1, 3] = c
    R[:, 2, 0] = c
    R[:, 2, 1] = d
    R[:, 2, 2] = a
    R[:, 2, 3] = -b
    R[:, 3, 0] = d
    R[:, 3, 1] = -c
    R[:, 3, 2] = b
    R[:, 3, 3] = a
    return R[0] if batch_dim_added else R


def quaternion_to_right_isoclinic(quaternions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # add batch dimension if not present
    if batch_dim_added := quaternions.dim() == 1:
        quaternions = quaternions[None]
    if normalize:
        quaternions = torch.nn.functional.normalize(quaternions)
    R = torch.empty((quaternions.shape[0], 4, 4), dtype=quaternions.dtype, device=quaternions.device)
    p, q, r, s = torch.unbind(quaternions, dim=1)

    R[:, 0, 0] = p
    R[:, 0, 1] = -q
    R[:, 0, 2] = -r
    R[:, 0, 3] = -s
    R[:, 1, 0] = q
    R[:, 1, 1] = p
    R[:, 1, 2] = s
    R[:, 1, 3] = -r
    R[:, 2, 0] = r
    R[:, 2, 1] = -s
    R[:, 2, 2] = p
    R[:, 2, 3] = q
    R[:, 3, 0] = s
    R[:, 3, 1] = r
    R[:, 3, 2] = -q
    R[:, 3, 3] = p
    return R[0] if batch_dim_added else R


def build_4d_rotation(quaternions_left: torch.Tensor, quaternions_right: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Constructs 4x4 rotation matrices from two quaternions representing a left/right isoclinic rotation respectively."""
    R_left = quaternion_to_left_isoclinic(quaternions_left, normalize=normalize)
    R_right = quaternion_to_right_isoclinic(quaternions_right, normalize=normalize)
    R = R_left @ R_right
    return R
