from typing import NamedTuple, Any
import torch
from torch.autograd.function import once_differentiable

from FasterGSFusedCudaBackend import _C


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
    current_mean_lr: float
    adam_step_count: int

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
            self.current_mean_lr,
            self.adam_step_count,
        )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        autograd_dummy: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        sh_coefficients_0: torch.Tensor,
        sh_coefficients_rest: torch.Tensor,
        moments_means: torch.Tensor,
        moments_scales: torch.Tensor,
        moments_rotations: torch.Tensor,
        moments_opacities: torch.Tensor,
        moments_sh_coefficients_0: torch.Tensor,
        moments_sh_coefficients_rest: torch.Tensor,
        densification_info: torch.Tensor,
        rasterizer_settings: RasterizerSettings,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        (
            image,
            primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
            n_instances, n_buckets, instance_primitive_indices_selector
        ) = _C.forward(
            means,
            scales,
            rotations,
            opacities,
            sh_coefficients_0,
            sh_coefficients_rest,
            *rasterizer_settings.as_tuple(),
        )
        ctx.rasterizer_settings = rasterizer_settings
        ctx.buffer_state = (n_instances, n_buckets, instance_primitive_indices_selector)
        ctx.save_for_backward(
            image,
            primitive_buffers,
            tile_buffers,
            instance_buffers,
            bucket_buffers,
        )
        ctx.means = means
        ctx.scales = scales
        ctx.rotations = rotations
        ctx.opacities = opacities
        ctx.sh_coefficients_0 = sh_coefficients_0
        ctx.sh_coefficients_rest = sh_coefficients_rest
        ctx.moments_means = moments_means
        ctx.moments_scales = moments_scales
        ctx.moments_rotations = moments_rotations
        ctx.moments_opacities = moments_opacities
        ctx.moments_sh_coefficients_0 = moments_sh_coefficients_0
        ctx.moments_sh_coefficients_rest = moments_sh_coefficients_rest
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(means)
        ctx.mark_non_differentiable(scales)
        ctx.mark_non_differentiable(rotations)
        ctx.mark_non_differentiable(opacities)
        ctx.mark_non_differentiable(sh_coefficients_0)
        ctx.mark_non_differentiable(sh_coefficients_rest)
        ctx.mark_non_differentiable(moments_means)
        ctx.mark_non_differentiable(moments_scales)
        ctx.mark_non_differentiable(moments_rotations)
        ctx.mark_non_differentiable(moments_opacities)
        ctx.mark_non_differentiable(moments_sh_coefficients_0)
        ctx.mark_non_differentiable(moments_sh_coefficients_rest)
        ctx.mark_non_differentiable(densification_info)
        return image, autograd_dummy

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_image: torch.Tensor,
        _,
    ) -> 'tuple[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]':
        _C.backward(
            ctx.densification_info,
            ctx.means,
            ctx.scales,
            ctx.rotations,
            ctx.opacities,
            ctx.sh_coefficients_0,
            ctx.sh_coefficients_rest,
            ctx.moments_means,
            ctx.moments_scales,
            ctx.moments_rotations,
            ctx.moments_opacities,
            ctx.moments_sh_coefficients_0,
            ctx.moments_sh_coefficients_rest,
            grad_image,
            *ctx.saved_tensors,
            *ctx.rasterizer_settings.as_tuple(),
            *ctx.buffer_state,
        )
        return (
            None,  # autograd_dummy
            None,  # means
            None,  # scales
            None,  # rotations
            None,  # opacities
            None,  # sh_coefficients_0
            None,  # sh_coefficients_rest
            None,  # moments_means
            None,  # moments_scales
            None,  # moments_rotations
            None,  # moments_opacities
            None,  # moments_sh_coefficients_0
            None,  # moments_sh_coefficients_rest
            None,  # densification_info
            None,  # rasterizer_settings
        )


def diff_rasterize(
    autograd_dummy: torch.Tensor,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients_0: torch.Tensor,
    sh_coefficients_rest: torch.Tensor,
    densification_info: torch.Tensor,
    rasterizer_settings: RasterizerSettings,
    moments_means: torch.Tensor = None,
    moments_scales: torch.Tensor = None,
    moments_rotations: torch.Tensor = None,
    moments_opacities: torch.Tensor = None,
    moments_sh_coefficients_0: torch.Tensor = None,
    moments_sh_coefficients_rest: torch.Tensor = None,
) -> torch.Tensor:
    return _Rasterize.apply(
        autograd_dummy,
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients_0,
        sh_coefficients_rest,
        torch.empty(0) if moments_means is None else moments_means,
        torch.empty(0) if moments_scales is None else moments_scales,
        torch.empty(0) if moments_rotations is None else moments_rotations,
        torch.empty(0) if moments_opacities is None else moments_opacities,
        torch.empty(0) if moments_sh_coefficients_0 is None else moments_sh_coefficients_0,
        torch.empty(0) if moments_sh_coefficients_rest is None else moments_sh_coefficients_rest,
        densification_info,
        rasterizer_settings,
    )
