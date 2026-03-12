"""FasterGS/Model.py"""

import math

import torch
import numpy as np

import Framework
from Cameras.Perspective import PerspectiveCamera
from CudaUtils.MortonEncoding import morton_encode
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.Base.Model import BaseModel
from Cameras.utils import quaternion_to_rotation_matrix
from Methods.FasterGS.FasterGSCudaBackend import FusedAdam, update_3d_filter, relocation_adjustment, add_noise
from Optim.adam_utils import replace_param_group_data, prune_param_groups, extend_param_groups, sort_param_groups, reset_state
from Optim.lr_utils import LRDecayPolicy
from Optim.knn_utils import compute_root_mean_squared_knn_distances


class Gaussians(torch.nn.Module):
    """Stores a set of 3D Gaussians."""

    def __init__(self, sh_degree: int, pretrained: bool) -> None:
        super().__init__()
        self.active_sh_degree = sh_degree if pretrained else 0
        self.active_sh_bases = (self.active_sh_degree + 1) ** 2
        self.max_sh_degree = sh_degree
        self.register_parameter('_means', None)
        self.register_parameter('_sh_coefficients_0', None)
        self.register_parameter('_sh_coefficients_rest', None)
        self.register_parameter('_scales', None)
        self.register_parameter('_rotations', None)
        self.register_parameter('_opacities', None)
        self._densification_info = None
        self.optimizer = None
        self.percent_dense = 0.0
        self.training_cameras_extent = 1.0
        self._filter_3d = None
        self.use_original_3d_filter = False
        self.use_optimized_3d_filter = False
        self.distance2filter = 0
        self.lr_means = 0.0
        self.lr_means_scheduler = None

    @property
    def means(self) -> torch.Tensor:
        """Returns the Gaussians' means (N, 3)."""
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales (N, 3)."""
        scales = self._scales.exp()
        if self.use_original_3d_filter:
            scales = (scales.square() + self._filter_3d).sqrt()
        return scales

    @property
    def raw_scales(self) -> torch.Tensor:
        """Returns the Gaussians' scales in logspace (N, 3)."""
        raw_scales = self._scales
        if self.use_original_3d_filter:
            scales = (raw_scales.exp().square() + self._filter_3d).sqrt()
            raw_scales = scales.log()
        return raw_scales

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
        opacities = self._opacities.sigmoid()
        if self.use_original_3d_filter:
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities = opacities * coef[..., None]
        return opacities

    @property
    def raw_opacities(self) -> torch.Tensor:
        """Returns the Gaussians' unactivated opacities (N, 1)."""
        raw_opacities = self._opacities
        if self.use_original_3d_filter:
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities = raw_opacities.sigmoid() * coef[..., None]
            raw_opacities = opacities.logit(eps=1e-6)
        return raw_opacities

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

    def opacity_regularization_loss(self) -> torch.Tensor:
        """Encourages the Gaussians' opacities to be small."""
        return self.opacities.mean()

    def scale_regularization_loss(self) -> torch.Tensor:
        """Encourages the Gaussians' scales to be small."""
        return self.scales.mean()

    def increase_used_sh_degree(self) -> None:
        """Increases the used SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.active_sh_bases = (self.active_sh_degree + 1) ** 2

    def setup_3d_filter(self, filter_config: Framework.ConfigParameterList, dataset: 'BaseDataset') -> None:
        """Sets up a 3D filter (see https://arxiv.org/abs/2311.16493)."""
        if filter_config.ORIGINAL_FORMULATION:
            self.use_original_3d_filter = True
            Logger.log_info(f'using mip-splatting 3d filter with variance {filter_config.FILTER_VARIANCE}')
        else:
            self.use_optimized_3d_filter = True
            Logger.log_info(f'using optimized 3d filter with variance {filter_config.FILTER_VARIANCE}')
        max_focal = 1e-12
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('update3dfilter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('update3dfilter ignores all distortion parameters')
            max_focal = max(max_focal, max(view.camera.focal_x, view.camera.focal_y))
        # assume max_focal is focal length of the highest resolution camera
        self.distance2filter = math.sqrt(filter_config.FILTER_VARIANCE) / max_focal
        self.compute_3d_filter(dataset)

    def compute_3d_filter(self, dataset: 'BaseDataset', clipping_tolerance: float = 0.15) -> None:
        """Computes the 3D filter."""
        positions = self.means
        filter_3d = torch.full((positions.shape[0], 1), fill_value=torch.finfo(torch.float32).max, device=positions.device, dtype=torch.float32)
        visibility_mask = torch.zeros((positions.shape[0], 1), device=positions.device, dtype=torch.bool)
        for view in dataset:
            if not isinstance(view.camera, PerspectiveCamera):
                raise Framework.ModelError('update3dfilter only supports perspective cameras')
            if view.camera.distortion is not None:
                Logger.log_warning('update3dfilter ignores all distortion parameters')
            update_3d_filter(
                positions,
                view.w2c,
                filter_3d,
                visibility_mask,
                view.camera.width,
                view.camera.height,
                view.camera.focal_x,
                view.camera.focal_y,
                view.camera.center_x,
                view.camera.center_y,
                view.camera.near_plane,
                clipping_tolerance,
                self.distance2filter,
            )
        filter_3d_max = filter_3d[visibility_mask].max()
        filter_3d = torch.where(visibility_mask, filter_3d, filter_3d_max, out=filter_3d)
        if self.use_original_3d_filter:
            filter_3d = filter_3d.square()  # original implementation always needs this in squared form
        elif self.use_optimized_3d_filter:
            filter_3d = filter_3d.log()  # optimized implementation uses this to directly clamp scales in logspace
        self._filter_3d = filter_3d

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, use_mcmc: bool) -> None:
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
        distances = distances * 0.1 if use_mcmc else distances
        scales = distances.log()[..., None].repeat(1, 3)
        # initial rotations
        rotations = torch.zeros((n_initial_gaussians, 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1.0
        # initial opacities
        initial_opacity = 0.5 if use_mcmc else 0.1
        initial_opacity_logit = math.log(initial_opacity / (1.0 - initial_opacity))
        opacities = torch.full((n_initial_gaussians, 1), fill_value=initial_opacity_logit, dtype=torch.float32, device='cuda')
        # setup parameters
        self._means = torch.nn.Parameter(means.contiguous())
        self._sh_coefficients_0 = torch.nn.Parameter(sh_coefficients_0.contiguous())
        self._sh_coefficients_rest = torch.nn.Parameter(sh_coefficients_rest.contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())

    def training_setup(self, training_wrapper, training_cameras_extent: float) -> None:
        """Sets up the optimizer."""
        self.percent_dense = training_wrapper.DENSIFICATION_PERCENT_DENSE
        self.training_cameras_extent = training_cameras_extent

        param_groups = [
            {'params': [self._means], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * self.training_cameras_extent, 'name': 'means'},
            {'params': [self._sh_coefficients_0], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_0, 'name': 'sh_coefficients_0'},
            {'params': [self._sh_coefficients_rest], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SH_COEFFICIENTS_REST, 'name': 'sh_coefficients_rest'},
            {'params': [self._opacities], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_OPACITIES, 'name': 'opacities'},
            {'params': [self._scales], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_SCALES, 'name': 'scales'},
            {'params': [self._rotations], 'lr': training_wrapper.OPTIMIZER.LEARNING_RATE_ROTATIONS, 'name': 'rotations'}
        ]

        self.optimizer = FusedAdam(param_groups, lr=0.0, eps=1e-15)

        self.lr_means_scheduler = LRDecayPolicy(
            lr_init=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_INIT * self.training_cameras_extent,
            lr_final=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_FINAL * self.training_cameras_extent,
            max_steps=training_wrapper.OPTIMIZER.LEARNING_RATE_MEANS_MAX_STEPS
        )

    def update_learning_rate(self, iteration: int) -> None:
        """Computes the current learning rate for the given iteration."""
        self.lr_means = self.lr_means_scheduler(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'means':
                param_group['lr'] = self.lr_means

    def reset_opacities(self) -> None:
        """Resets the opacities to a fixed value."""
        opacities_new = self._opacities.clamp_max(-4.595119953155518)  # sigmoid(-4.595119953155518) = 0.01
        if self.use_original_3d_filter:
            # make sure that the current 3d filter has the same effect on the new opacities
            scales_square = self._scales.exp().square()
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + self._filter_3d
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            opacities_new = (opacities_new.sigmoid() / coef[..., None]).logit(eps=1e-6)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    def prune(self, prune_mask: torch.Tensor) -> None:
        """Prunes Gaussians that are not visible or too large."""
        valid_mask = ~prune_mask
        param_groups = prune_param_groups(self.optimizer, valid_mask)

        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, valid_mask].contiguous()
        if self._filter_3d is not None:
            self._filter_3d = self._filter_3d[valid_mask].contiguous()

    def sort(self, ordering: torch.Tensor) -> None:
        """Applies the given ordering to the Gaussians."""
        param_groups = sort_param_groups(self.optimizer, ordering)

        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']

        if self._densification_info is not None:
            self._densification_info = self._densification_info[:, ordering].contiguous()
        if self._filter_3d is not None:
            self._filter_3d = self._filter_3d[ordering].contiguous()

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
        param_groups = extend_param_groups(self.optimizer, {
            'means': torch.cat([duplicated_means, split_means]),
            'sh_coefficients_0': torch.cat([duplicated_sh_coefficients_0, split_sh_coefficients_0]),
            'sh_coefficients_rest': torch.cat([duplicated_sh_coefficients_rest, split_sh_coefficients_rest]),
            'opacities': torch.cat([duplicated_opacities, split_opacities]),
            'scales': torch.cat([duplicated_scales, split_scales]),
            'rotations': torch.cat([duplicated_rotations, split_rotations])
        })
        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']

        # if they were set, densification info and 3d filter are now no longer valid
        self._densification_info = None
        self._filter_3d = None

        # prune
        prune_mask = torch.cat([split_mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device='cuda')])
        prune_mask |= self._opacities.flatten() < math.log(min_opacity / (1 - min_opacity))
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        if prune_large_gaussians:
            prune_mask |= self._scales.max(dim=1).values > math.log(0.1 * self.training_cameras_extent)
        self.prune(prune_mask)

    def mcmc_densification(self, min_opacity: float, cap_max: int) -> None:
        """Relocates low-opacity/degenerate Gaussians and adds new ones up to a cap."""
        # relocate
        dead_mask = self._opacities.flatten() <= math.log(min_opacity / (1 - min_opacity))
        dead_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        n_dead_gaussians = dead_mask.sum().item()
        if n_dead_gaussians > 0:
            # sample existing Gaussians to copy to the dead ones, with probability proportional to opacity
            dead_indices = torch.where(dead_mask)[0]
            alive_indices = torch.where(~dead_mask)[0]
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities[alive_indices], n_dead_gaussians, replacement=True)
            sampled_indices = alive_indices[sampled_indices]

            # compute the adjusted opacities and scales
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1  # +1 for the original Gaussian
            adjusted_opacities, adjusted_scales = relocation_adjustment(
                opacities[sampled_indices],
                self._scales[sampled_indices].exp(),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales.log()

            # update existing sampled Gaussians
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales

            # copy sampled Gaussians to the dead ones
            self._means[dead_indices] = self._means[sampled_indices]
            self._sh_coefficients_0[dead_indices] = self._sh_coefficients_0[sampled_indices]
            self._sh_coefficients_rest[dead_indices] = self._sh_coefficients_rest[sampled_indices]
            self._opacities[dead_indices] = adjusted_opacities
            self._scales[dead_indices] = adjusted_scales
            self._rotations[dead_indices] = self._rotations[sampled_indices]

            # reset optimizer state for the sampled Gaussians
            reset_state(self.optimizer, indices=sampled_indices)

            # if they were set, densification info and 3d filter are now no longer valid
            self._densification_info = None
            self._filter_3d = None

        # add new Gaussians
        current_n_points = self._means.shape[0]
        n_target = min(cap_max, int(1.05 * current_n_points))
        n_added_gaussians = max(0, n_target - current_n_points)
        if n_added_gaussians > 0:
            # sample existing Gaussians to duplicate, with probability proportional to opacity
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities, n_added_gaussians, replacement=True)

            # compute the adjusted opacities and scales
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1  # +1 for the original Gaussian
            adjusted_opacities, adjusted_scales = relocation_adjustment(
                opacities[sampled_indices],
                self._scales[sampled_indices].exp(),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales.log()

            # update existing sampled Gaussians
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales

            # add new Gaussians by duplicating the sampled ones
            param_groups = extend_param_groups(self.optimizer, {
                'means': self._means[sampled_indices],
                'sh_coefficients_0': self._sh_coefficients_0[sampled_indices],
                'sh_coefficients_rest': self._sh_coefficients_rest[sampled_indices],
                'opacities': adjusted_opacities,
                'scales': adjusted_scales,
                'rotations': self._rotations[sampled_indices],
            })
            self._means = param_groups['means']
            self._sh_coefficients_0 = param_groups['sh_coefficients_0']
            self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
            self._opacities = param_groups['opacities']
            self._scales = param_groups['scales']
            self._rotations = param_groups['rotations']

            # reset optimizer state for the sampled Gaussians
            reset_state(self.optimizer, indices=sampled_indices)

            # if they were set, densification info and 3d filter are now no longer valid
            self._densification_info = None
            self._filter_3d = None

    def apply_morton_ordering(self) -> None:
        """Applies Morton ordering to the Gaussians."""
        morton_encoding = morton_encode(self._means.data)
        order = torch.argsort(morton_encoding)
        self.sort(order)

    def importance_pruning(self, scores: torch.Tensor, pruning_ratio: float) -> None:
        """Prunes the given percentage of Gaussians with the lowest importance score (from Speedy-Splat)."""
        k = int(pruning_ratio * (scores.numel() - 1)) + 1  # kthvalue is 1-based
        threshold = torch.kthvalue(scores, k).values
        prune_mask = scores <= threshold
        self.prune(prune_mask)

    @torch.no_grad()
    def post_optimizer_step(self, inject_noise: bool) -> None:
        """Applies modifications to the Gaussians after every optimizer step."""
        if inject_noise:
            add_noise(self.raw_scales, self.raw_rotations, self.raw_opacities, self.means, 5e5 * self.lr_means)
        if self.use_optimized_3d_filter:
            self._scales.clamp_min_(self._filter_3d)

    @torch.no_grad()
    def training_cleanup(self, min_opacity: float) -> int:
        """Cleans the model after training."""
        # bake 3d filter if used
        if self.use_optimized_3d_filter:
            # nothing to do, already baked in
            self.use_optimized_3d_filter = False
        elif self.use_original_3d_filter:
            # the 3d filter must be baked into the opacities before the scales to get the correct result
            self._opacities.data = self.raw_opacities
            self._scales.data = self.raw_scales
            self.use_original_3d_filter = False
        self._filter_3d = None

        # densification info no longer needed
        self._densification_info = None

        # prune low-opacity and degenerate Gaussians
        prune_mask = self.opacities.flatten() < min_opacity
        prune_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        self.prune(prune_mask)

        # sort by morton code
        self.apply_morton_ordering()

        # clear any leftover gradients and delete optimizer
        self.optimizer.zero_grad()
        self.optimizer = None

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
class FasterGSModel(BaseModel):
    """Defines the FasterGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)
        self.gaussians: Gaussians | None = None

    def build(self) -> 'FasterGSModel':
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
        splat_render_mode = 'mip-0.1' if Framework.config.RENDERER.PROPER_ANTIALIASING else 'default'
        data['comments'] = [f'SplatRenderMode: {splat_render_mode}', 'Generated with NeRFICG/FasterGS']

        return data
