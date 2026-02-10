"""FasterGSFused/Model.py"""

import math

import torch
import numpy as np

import Framework
from CudaUtils.MortonEncoding import morton_encode
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.Model import BaseModel
from Cameras.utils import quaternion_to_rotation_matrix
from Optim.lr_utils import LRDecayPolicy
from Optim.knn_utils import compute_root_mean_squared_knn_distances


class Gaussians(torch.nn.Module):
    """Stores a set of 3D Gaussians."""

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_buffer('_means', None)
        self.register_buffer('_sh_coefficients_0', None)
        self.register_buffer('_sh_coefficients_rest', None)
        self.register_buffer('_scales', None)
        self.register_buffer('_rotations', None)
        self.register_buffer('_opacities', None)
        self._densification_info = None
        self.percent_dense = 0.0
        self.training_cameras_extent = 1.0
        self.lr_means = 0.0
        self.lr_means_scheduler = None
        # adam moments
        self.moments_means = torch.empty(0)
        self.moments_sh_coefficients_0 = torch.empty(0)
        self.moments_sh_coefficients_rest = torch.empty(0)
        self.moments_scales = torch.empty(0)
        self.moments_rotations = torch.empty(0)
        self.moments_opacities = torch.empty(0)

    @property
    def means(self) -> torch.Tensor:
        """Returns the Gaussians' means (N, 3)."""
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales (N, 3)."""
        return self._scales.exp()

    @property
    def raw_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales in logspace (N, 3)."""
        return self._scales

    @property
    def rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as quaternions (N, 4)."""
        return torch.nn.functional.normalize(self._rotations)

    @property
    def raw_rotations(self) -> torch.Tensor:
        """Returns the Gaussians' rotations as unnormalized quaternions (N, 4)."""
        return self._rotations

    @property
    def opacities(self) -> torch.Tensor:
        """Returns the Gaussians' opacities (N, 1)."""
        return self._opacities.sigmoid()

    @property
    def raw_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' unactivated opacities (N, 1)."""
        return self._opacities

    @property
    def sh_coefficients(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for all bases (N, (max_degree + 1) ** 2, 3)."""
        return torch.cat([self._sh_coefficients_0, self._sh_coefficients_rest], dim=1)

    @property
    def sh_coefficients_0(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for the 0th, view-independent basis (N, 1, 3)."""
        return self._sh_coefficients_0

    @property
    def sh_coefficients_rest(self) -> torch.Tensor:
        """Returns the Gaussians' SH coefficients for all view-dependent bases (N, (max_degree + 1) ** 2 - 1, 3)."""
        return self._sh_coefficients_rest

    @property
    def densification_info(self) -> torch.Tensor:
        """Returns the current densification info buffers (2, N)."""
        return self._densification_info

    @property
    def covariances(self) -> torch.Tensor:
        """Returns the Gaussians' covariance matrices (N, 3, 3)."""
        R = quaternion_to_rotation_matrix(self.rotations, normalize=False)
        S = torch.diag_embed(self.scales)
        RS = R @ S
        return RS @ RS.transpose(-2, -1)

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud) -> None:
        """Initializes the model from a point cloud."""
        # initial means
        means = point_cloud.positions.cuda()
        n_initial_gaussians = means.shape[0]
        Logger.log_info(f'number of Gaussians at initialization: {n_initial_gaussians:,}')
        # initial sh coefficients
        rgbs = torch.full_like(means, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        sh_coefficients_0 = ((rgbs - 0.5) / 0.28209479177387814)[:, None, :]
        sh_coefficients_rest = torch.zeros((n_initial_gaussians, (self.max_sh_degree + 1) ** 2 - 1, 3), dtype=torch.float32, device='cuda')
        # initial scales
        distances = compute_root_mean_squared_knn_distances(means)
        scales = distances.log()[..., None].repeat(1, 3)
        # initial rotations
        rotations = torch.zeros((n_initial_gaussians, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0
        # initial opacities
        initial_opacity = 0.1
        initial_opacity_logit = math.log(initial_opacity / (1.0 - initial_opacity))
        opacities = torch.full((n_initial_gaussians, 1), fill_value=initial_opacity_logit, dtype=torch.float32, device='cuda')
        # setup buffers
        self._means = means.contiguous()
        self._sh_coefficients_0 = sh_coefficients_0.contiguous()
        self._sh_coefficients_rest = sh_coefficients_rest.contiguous()
        self._scales = scales.contiguous()
        self._rotations = rotations.contiguous()
        self._opacities = opacities.contiguous()
        # setup adam moments
        self.moments_means = torch.zeros(*self.means.shape, 2, dtype=torch.float32, device='cuda')
        self.moments_sh_coefficients_0 = torch.zeros(*self._sh_coefficients_0.shape, 2, dtype=torch.float32, device='cuda')
        self.moments_sh_coefficients_rest = torch.zeros(*self._sh_coefficients_rest.shape, 2, dtype=torch.float32, device='cuda')
        self.moments_scales = torch.zeros(*self._scales.shape, 2, dtype=torch.float32, device='cuda')
        self.moments_rotations = torch.zeros(*self._rotations.shape, 2, dtype=torch.float32, device='cuda')
        self.moments_opacities = torch.zeros(*self._opacities.shape, 2, dtype=torch.float32, device='cuda')

    def training_setup(self, training_wrapper, training_cameras_extent: float) -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.DENSIFICATION_PERCENT_DENSE
        self.training_cameras_extent = training_cameras_extent

        self.lr_means = training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * self.training_cameras_extent
        self.lr_means_scheduler = LRDecayPolicy(
            lr_init=self.lr_means,
            lr_final=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_FINAL * self.training_cameras_extent,
            max_steps=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_MAX_STEPS
        )

    def update_learning_rate(self, iteration: int) -> None:
        """Computes the current learning rate for the given iteration."""
        self.lr_means = self.lr_means_scheduler(iteration)

    def reset_opacities(self) -> None:
        """Resets the opacities to a fixed value."""
        self._opacities.clamp_max_(-4.595119953155518)  # sigmoid(-4.595119953155518) = 0.01
        self.moments_opacities.zero_()

    def prune(self, prune_mask: torch.Tensor) -> None:
        """Prunes Gaussians that are not visible or too large."""
        valid_mask = ~prune_mask

        self._means.data = self._means[valid_mask].contiguous()
        self._sh_coefficients_0.data = self._sh_coefficients_0[valid_mask].contiguous()
        self._sh_coefficients_rest.data = self._sh_coefficients_rest[valid_mask].contiguous()
        self._opacities.data = self._opacities[valid_mask].contiguous()
        self._scales.data = self._scales[valid_mask].contiguous()
        self._rotations.data = self._rotations[valid_mask].contiguous()

        self.moments_means = self.moments_means[valid_mask].contiguous()
        self.moments_sh_coefficients_0 = self.moments_sh_coefficients_0[valid_mask].contiguous()
        self.moments_sh_coefficients_rest = self.moments_sh_coefficients_rest[valid_mask].contiguous()
        self.moments_opacities = self.moments_opacities[valid_mask].contiguous()
        self.moments_scales = self.moments_scales[valid_mask].contiguous()
        self.moments_rotations = self.moments_rotations[valid_mask].contiguous()

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, valid_mask].contiguous()

    def sort(self, ordering: torch.Tensor) -> None:
        """Applies the given ordering to the Gaussians."""
        self._means.data = self._means[ordering].contiguous()
        self._sh_coefficients_0.data = self._sh_coefficients_0[ordering].contiguous()
        self._sh_coefficients_rest.data = self._sh_coefficients_rest[ordering].contiguous()
        self._opacities.data = self._opacities[ordering].contiguous()
        self._scales.data = self._scales[ordering].contiguous()
        self._rotations.data = self._rotations[ordering].contiguous()

        self.moments_means = self.moments_means[ordering].contiguous()
        self.moments_sh_coefficients_0 = self.moments_sh_coefficients_0[ordering].contiguous()
        self.moments_sh_coefficients_rest = self.moments_sh_coefficients_rest[ordering].contiguous()
        self.moments_opacities = self.moments_opacities[ordering].contiguous()
        self.moments_scales = self.moments_scales[ordering].contiguous()
        self.moments_rotations = self.moments_rotations[ordering].contiguous()

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, ordering].contiguous()

    def reset_densification_info(self):
        self._densification_info = torch.zeros((2, self._means.shape[0]), dtype=torch.float32, device='cuda')

    def adaptive_density_control(self, grad_threshold: float, min_opacity: float, prune_large_gaussians: bool) -> None:
        """Densify Gaussians and prune those that are not visible or too large."""
        densification_mask = self.densification_info[1] >= grad_threshold * self.densification_info[0].clamp_min(1.0)
        is_small = torch.max(self._scales, dim=1).values <= math.log(self.percent_dense * self.training_cameras_extent)

        # duplicate small gaussians
        duplicate_mask = densification_mask & is_small
        n_new_gaussians_duplicate = duplicate_mask.sum().item()
        duplicated_means = self._means[duplicate_mask]
        duplicated_sh_coefficients_0 = self._sh_coefficients_0[duplicate_mask]
        duplicated_sh_coefficients_rest = self._sh_coefficients_rest[duplicate_mask]
        duplicated_opacities = self._opacities[duplicate_mask]
        duplicated_scales = self._scales[duplicate_mask]
        duplicated_rotations = self._rotations[duplicate_mask]

        # split large gaussians
        split_mask = densification_mask & ~is_small
        n_new_gaussians_split = 2 * split_mask.sum().item()
        split_scales = self._scales[split_mask].exp().expand(2, -1, -1).flatten(end_dim=1)
        split_rotations = self._rotations[split_mask].expand(2, -1, -1).flatten(end_dim=1)
        offsets = (quaternion_to_rotation_matrix(split_rotations) @ (split_scales * torch.randn_like(split_scales))[..., None])[..., 0]
        split_means = self._means[split_mask].expand(2, -1, -1).flatten(end_dim=1) + offsets
        split_scales = split_scales.mul(0.625).log()  # 1 / 1.6 = 0.625
        split_sh_coefficients_0 = self._sh_coefficients_0[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_sh_coefficients_rest = self._sh_coefficients_rest[split_mask].expand(2, -1, -1, -1).flatten(end_dim=1)
        split_opacities = self._opacities[split_mask].expand(2, -1, -1).flatten(end_dim=1)

        # incorporate
        n_new_gaussians = n_new_gaussians_duplicate + n_new_gaussians_split
        self._means.data = torch.cat([self._means, duplicated_means, split_means])
        self._sh_coefficients_0.data = torch.cat([self._sh_coefficients_0, duplicated_sh_coefficients_0, split_sh_coefficients_0])
        self._sh_coefficients_rest.data = torch.cat([self._sh_coefficients_rest, duplicated_sh_coefficients_rest, split_sh_coefficients_rest])
        self._opacities.data = torch.cat([self._opacities, duplicated_opacities, split_opacities])
        self._scales.data = torch.cat([self._scales, duplicated_scales, split_scales])
        self._rotations.data = torch.cat([self._rotations, duplicated_rotations, split_rotations])
        self.moments_means = torch.cat([self.moments_means, torch.zeros((n_new_gaussians, *self.moments_means.shape[1:]), dtype=torch.float32, device='cuda')])
        self.moments_sh_coefficients_0 = torch.cat([self.moments_sh_coefficients_0, torch.zeros((n_new_gaussians, *self.moments_sh_coefficients_0.shape[1:]), dtype=torch.float32, device='cuda')])
        self.moments_sh_coefficients_rest = torch.cat([self.moments_sh_coefficients_rest, torch.zeros((n_new_gaussians, *self.moments_sh_coefficients_rest.shape[1:]), dtype=torch.float32, device='cuda')])
        self.moments_opacities = torch.cat([self.moments_opacities, torch.zeros((n_new_gaussians, *self.moments_opacities.shape[1:]), dtype=torch.float32, device='cuda')])
        self.moments_scales = torch.cat([self.moments_scales, torch.zeros((n_new_gaussians, *self.moments_scales.shape[1:]), dtype=torch.float32, device='cuda')])
        self.moments_rotations = torch.cat([self.moments_rotations, torch.zeros((n_new_gaussians, *self.moments_rotations.shape[1:]), dtype=torch.float32, device='cuda')])

        # if it was set, densification info is now no longer valid
        self._densification_info = None

        # prune
        prune_mask = torch.cat([split_mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device='cuda')])
        prune_mask |= self._opacities.flatten() < math.log(min_opacity / (1 - min_opacity))
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        if prune_large_gaussians:
            prune_mask |= self._scales.max(dim=1).values > math.log(0.1 * self.training_cameras_extent)
        self.prune(prune_mask)

    def apply_morton_ordering(self) -> None:
        """Applies Morton ordering to the Gaussians."""
        morton_encoding = morton_encode(self._means.data)
        order = torch.argsort(morton_encoding)
        self.sort(order)

    @torch.no_grad()
    def training_cleanup(self, min_opacity: float) -> int:
        """Cleans the model after training."""
        # densification info no longer needed
        self._densification_info = None

        # prune low-opacity and degenerate Gaussians
        prune_mask = self.opacities.flatten() < min_opacity
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        self.prune(prune_mask)

        # sort by morton code
        self.apply_morton_ordering()

        # moments no longer needed
        self.moments_means = None
        self.moments_sh_coefficients_0 = None
        self.moments_sh_coefficients_rest = None
        self.moments_opacities = None
        self.moments_scales = None
        self.moments_rotations = None

        return self.means.shape[0]

    @torch.no_grad()
    def as_ply_dict(self) -> dict[str, np.ndarray]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        if self.means.shape[0] == 0:
            return {}

        # construct attributes
        means = self.means.detach().contiguous().cpu().numpy()
        sh_0 = self.sh_coefficients_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self.sh_coefficients_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.raw_opacities.detach().contiguous().cpu().numpy()  # most viewers expect unactivated opacities
        scales = self.raw_scales.detach().contiguous().cpu().numpy()  # most viewers expect unactivated scales
        rotations = self.rotations.detach().contiguous().cpu().numpy()
        attributes = np.concatenate((means, sh_0, sh_rest, opacities, scales, rotations), axis=1)

        # construct structured array
        attribute_names = (
              ['x', 'y', 'z']                                    # 3d mean
            + ['f_dc_0', 'f_dc_1', 'f_dc_2']                     # 0-th SH degree coefficients
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]  # remaining SH degree coefficients
            + ['opacity']                                        # opacity (pre-activation)
            + ['scale_0', 'scale_1', 'scale_2']                  # 3d scale (pre-activation)
            + ['rot_0', 'rot_1', 'rot_2', 'rot_3']               # rotation quaternion
        )
        dtype = 'f4'  # store all attributes as float32 for compatibility
        full_dtype = [(attribute_name, dtype) for attribute_name in attribute_names]
        vertices = np.empty(means.shape[0], dtype=full_dtype)

        # insert attributes into structured array
        vertices[:] = list(map(tuple, attributes))

        return {'vertex': vertices}


@Framework.Configurable.configure(
    SH_DEGREE=3,
)
class FasterGSFusedModel(BaseModel):
    """Defines the FasterGSFused model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None

    def build(self) -> 'FasterGSFusedModel':
        """Builds the model."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians(self.SH_DEGREE, pretrained)
        return self

    def get_ply_dict(self) -> dict[str, np.ndarray | list[str]]:
        """Returns the model as a ply-compatible dictionary using structured numpy arrays."""
        data: dict[str, np.ndarray | list[str]] = {}
        if self.gaussians is None or not (data := self.gaussians.as_ply_dict()):
            return data

        # add method-specific comments
        data['comments'] = ['SplatRenderMode: default', 'Generated with NeRFICG/FasterGSFused']

        return data
