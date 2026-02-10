"""FasterGS4D/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGS4D.Model import FasterGS4DModel
from Methods.FasterGS4D.FasterGS4DCudaBackend import diff_rasterize, RasterizerSettings


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGS4D renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        bg_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
        view.timestamp,
    )


@Framework.Configurable.configure(
    SCALE_MODIFIER=1.0,
)
class FasterGS4DRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGS4DModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGS4D renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGS4D renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Renders an image for a given view."""
        densification_info = torch.zeros_like(self.model.gaussians.densification_info) if update_densification_info else torch.empty(0)
        image = diff_rasterize(
            spatial_means=self.model.gaussians.spatial_means,
            temporal_means=self.model.gaussians.temporal_means,
            spatial_scales=self.model.gaussians.raw_spatial_scales,
            temporal_scales=self.model.gaussians.raw_temporal_scales,
            left_isoclinic_rotations=self.model.gaussians.raw_left_isoclinic_rotations,
            right_isoclinic_rotations=self.model.gaussians.raw_right_isoclinic_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=densification_info,
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, bg_color),
        )
        return image, densification_info

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = diff_rasterize(
            spatial_means=self.model.gaussians.spatial_means,
            temporal_means=self.model.gaussians.temporal_means,
            spatial_scales=self.model.gaussians.raw_spatial_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            temporal_scales=self.model.gaussians.raw_temporal_scales + math.log(max(self.SCALE_MODIFIER, 1e-6)),
            left_isoclinic_rotations=self.model.gaussians.raw_left_isoclinic_rotations,
            right_isoclinic_rotations=self.model.gaussians.raw_right_isoclinic_rotations,
            opacities=self.model.gaussians.raw_opacities,
            sh_coefficients_0=self.model.gaussians.sh_coefficients_0,
            sh_coefficients_rest=self.model.gaussians.sh_coefficients_rest,
            densification_info=torch.empty(0),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases, view.camera.background_color),
        )
        image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
