from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSTestbedCudaBackend import _C


class RasterizerSettings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    bg_color: torch.Tensor  # background color in RGB format
    active_sh_bases: int  # number of spherical harmonics bases to use for color computation
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    use_tight_bounds: bool  # whether to use tight rectangular bounding boxes
    use_opacity_based_bounds: bool  # whether tight bounding boxes should consider opacity
    use_fused_activations: bool  # whether to apply activations within the rasterizer
    use_load_balanced_instance_creation: bool  # whether to use load-balanced instance creation
    use_tile_based_culling: bool  # whether to use tile-based culling
    use_per_gaussian_backward: bool  # whether to use per-Gaussian backward pass
    use_separate_sorting: bool  # whether to split up sorting into separate steps for depth and tile

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.bg_color,
            self.active_sh_bases,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.use_tight_bounds,
            self.use_opacity_based_bounds,
            self.use_fused_activations,
            self.use_load_balanced_instance_creation,
            self.use_tile_based_culling,
            self.use_per_gaussian_backward,
            self.use_separate_sorting,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> torch.Tensor:
        (
            image,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            means,
            scales,
            rotations,
            sh_coefficients,
            sh_coefficients_rest,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
    ) -> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None]':
        (
            grad_means, grad_scales, grad_rotations, grad_opacities,
            grad_sh_coefficients, grad_sh_coefficients_0, grad_sh_coefficients_rest
        ) = _C.backward(
            ctx.densification_info,
            grad_image,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            grad_means,
            grad_scales,
            grad_rotations,
            grad_opacities,
            grad_sh_coefficients,
            grad_sh_coefficients_0,
            grad_sh_coefficients_rest,
            None,  # densification_info
            None,  # rasterizer_settings
        )


def diff_rasterize(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
) -> torch.Tensor:
    return _Rasterize.apply(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        sh_coefficients_0,
        sh_coefficients_rest,
        densification_info,
        rasterizer_settings,
    )
