"""FasterGSBasis/Renderer.py"""

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSBasis.Model import FasterGSBasisModel
from Methods.FasterGSBasis.FasterGSBasisCudaBackend import diff_rasterize, RasterizerSettings


def extract_settings(view: View, active_sh_bases: int) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGSBasis renderer only supports perspective cameras')
    if view.camera.distortion is not None:
        Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
    return RasterizerSettings(
        view.w2c,
        view.position,
        view.camera.background_color,
        active_sh_bases,
        view.camera.width,
        view.camera.height,
        view.camera.focal_x,
        view.camera.focal_y,
        view.camera.center_x,
        view.camera.center_y,
        view.camera.near_plane,
        view.camera.far_plane,
    )


@Framework.Configurable.configure(
    CLAMP_IMAGE_TRAINING=False,
    SCALE_MODIFIER=1.0,
)
class FasterGSBasisRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSBasisModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGSBasis renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGSBasis renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool) -> torch.Tensor:
        """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.scales,
            rotations=self.model.gaussians.rotations,
            opacities=self.model.gaussians.opacities,
            sh_coefficients=self.model.gaussians.sh_coefficients,
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases)
        )
        if self.CLAMP_IMAGE_TRAINING:
            image = image.clamp(0.0, 1.0)
        return image

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=self.model.gaussians.scales * self.SCALE_MODIFIER,
            rotations=self.model.gaussians.rotations,
            opacities=self.model.gaussians.opacities,
            sh_coefficients=self.model.gaussians.sh_coefficients,
            densification_info=torch.empty(0),
            rasterizer_settings=extract_settings(view, self.model.gaussians.active_sh_bases)
        ).clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
