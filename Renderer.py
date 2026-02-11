"""FasterGSTestbed/Renderer.py"""

import math

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.FasterGSTestbed.Model import FasterGSTestbedModel
from Methods.FasterGSTestbed.FasterGSTestbedCudaBackend import diff_rasterize, RasterizerSettings


def extract_settings(
    view: View,
    active_sh_bases: int,
    bg_color: torch.Tensor,
    use_tight_bounds: bool,
    use_opacity_based_bounds: bool,
    use_fused_activations: bool,
    use_load_balanced_instance_creation: bool,
    use_tile_based_culling: bool,
    use_per_gaussian_backward: bool,
    use_separate_sorting: bool,
) -> RasterizerSettings:
    if not isinstance(view.camera, PerspectiveCamera):
        raise Framework.RendererError('FasterGSTestbed renderer only supports perspective cameras')
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
        use_tight_bounds,
        use_opacity_based_bounds,
        use_fused_activations,
        use_load_balanced_instance_creation,
        use_tile_based_culling,
        use_per_gaussian_backward,
        use_separate_sorting,
    )


@Framework.Configurable.configure(
    CONCATENATE_SH_COEFFICIENTS=True,
    USE_TIGHT_BOUNDS=False,
    USE_OPACITY_BASED_BOUNDS=False,
    USE_FUSED_ACTIVATIONS=False,
    USE_LOAD_BALANCED_INSTANCE_CREATION=False,
    USE_TILE_BASED_CULLING=False,
    USE_PER_GAUSSIAN_BACKWARD=False,
    USE_SEPARATE_SORTING=False,
    SCALE_MODIFIER=1.0,
)
class FasterGSTestbedRenderer(BaseRenderer):
    """Wrapper around the rasterization module from 3DGS."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [FasterGSTestbedModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('FasterGSTestbed renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'FasterGSTestbed renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')

    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.model.training:
            raise Framework.RendererError('please directly call render_image_training() instead of render_image() during training')
        else:
            return self.render_image_inference(view, to_chw)

    def render_image_training(self, view: View, update_densification_info: bool, bg_color: torch.Tensor) -> torch.Tensor:
        """Renders an image for a given view."""
        if self.CONCATENATE_SH_COEFFICIENTS:
            sh_coefficients = self.model.gaussians.sh_coefficients
            sh_coefficients_0 = torch.empty(0)
            sh_coefficients_rest = torch.empty(0)
        else:
            sh_coefficients = torch.empty(0)
            sh_coefficients_0 = self.model.gaussians.sh_coefficients_0
            sh_coefficients_rest = self.model.gaussians.sh_coefficients_rest
        if self.USE_FUSED_ACTIVATIONS:
            scales = self.model.gaussians.raw_scales
            rotations = self.model.gaussians.raw_rotations
            opacities = self.model.gaussians.raw_opacities
        else:
            scales = self.model.gaussians.scales
            rotations = self.model.gaussians.rotations
            opacities = self.model.gaussians.opacities
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients=sh_coefficients,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=self.model.gaussians.densification_info if update_densification_info else torch.empty(0),
            rasterizer_settings=extract_settings(
                view, self.model.gaussians.active_sh_bases, bg_color,
                self.USE_TIGHT_BOUNDS, self.USE_OPACITY_BASED_BOUNDS, self.USE_FUSED_ACTIVATIONS,
                self.USE_LOAD_BALANCED_INSTANCE_CREATION, self.USE_TILE_BASED_CULLING,
                self.USE_PER_GAUSSIAN_BACKWARD, self.USE_SEPARATE_SORTING
            ),
        )
        return image

    @torch.no_grad()
    def render_image_inference(self, view: View, to_chw: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if self.CONCATENATE_SH_COEFFICIENTS:
            sh_coefficients = self.model.gaussians.sh_coefficients
            sh_coefficients_0 = torch.empty(0)
            sh_coefficients_rest = torch.empty(0)
        else:
            sh_coefficients = torch.empty(0)
            sh_coefficients_0 = self.model.gaussians.sh_coefficients_0
            sh_coefficients_rest = self.model.gaussians.sh_coefficients_rest
        safe_scale_modifier = max(self.SCALE_MODIFIER, 1e-6)
        if self.USE_FUSED_ACTIVATIONS:
            scales = self.model.gaussians.raw_scales + math.log(safe_scale_modifier)
            rotations = self.model.gaussians.raw_rotations
            opacities = self.model.gaussians.raw_opacities
        else:
            scales = self.model.gaussians.scales * safe_scale_modifier
            rotations = self.model.gaussians.rotations
            opacities = self.model.gaussians.opacities
        image = diff_rasterize(
            means=self.model.gaussians.means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coefficients=sh_coefficients,
            sh_coefficients_0=sh_coefficients_0,
            sh_coefficients_rest=sh_coefficients_rest,
            densification_info=torch.empty(0),
            rasterizer_settings=extract_settings(
                view, self.model.gaussians.active_sh_bases, view.camera.background_color,
                self.USE_TIGHT_BOUNDS, self.USE_OPACITY_BASED_BOUNDS, self.USE_FUSED_ACTIVATIONS,
                self.USE_LOAD_BALANCED_INSTANCE_CREATION, self.USE_TILE_BASED_CULLING,
                self.USE_PER_GAUSSIAN_BACKWARD, self.USE_SEPARATE_SORTING
            ),
        )
        image = image.clamp(0.0, 1.0)
        return {'rgb': image if to_chw else image.permute(1, 2, 0)}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {'rgb': outputs['rgb']}
