"""FasterGSBasis/Trainer.py"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import BasicPointCloud, apply_background_color
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.FasterGSBasis.Loss import FasterGSBasisLoss
from Methods.FasterGSBasis.utils import enable_expandable_segments
from Optim.Samplers.DatasetSamplers import DatasetSampler


@Framework.Configurable.configure(
    NUM_ITERATIONS=30_000,
    DENSIFICATION_START_ITERATION=600,  # while official code states 500, densification actually starts at 600 there
    DENSIFICATION_END_ITERATION=14_900,  # while official code states 15000, densification actually stops at 14900 there
    DENSIFICATION_INTERVAL=100,
    DENSIFICATION_GRAD_THRESHOLD=0.0002,
    DENSIFICATION_PERCENT_DENSE=0.01,
    OPACITY_RESET_INTERVAL=3_000,
    EXTRA_OPACITY_RESET_ITERATION=500,
    MIN_OPACITY_AFTER_TRAINING=1 / 255,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,  # weight for the per-pixel L1 loss on the rgb image
        LAMBDA_DSSIM=0.2,  # weight for the DSSIM loss on the rgb image
    ),
    OPTIMIZER=Framework.ConfigParameterList(
        LEARNING_RATE_MEANS_INIT=0.00016,
        LEARNING_RATE_MEANS_FINAL=0.0000016,
        LEARNING_RATE_MEANS_MAX_STEPS=30_000,
        LEARNING_RATE_SH_COEFFICIENTS_0=0.0025,
        LEARNING_RATE_SH_COEFFICIENTS_REST=0.000125,  # 0.0025 / 20
        LEARNING_RATE_OPACITIES=0.025,  # recently updated in official code; used to be 0.05
        LEARNING_RATE_SCALES=0.005,
        LEARNING_RATE_ROTATIONS=0.001,
    ),
)
class FasterGSBasisTrainer(GuiTrainer):
    """Defines the trainer for the FasterGSBasis variant."""

    def __init__(self, **kwargs) -> None:
        self.requires_empty_cache = True
        if not Framework.config.TRAINING.GUI.ACTIVATE:
            if enable_expandable_segments():
                self.requires_empty_cache = False
                Logger.log_info('using "expandable_segments:True" with the torch cuda memory allocator')
        super().__init__(**kwargs)
        self.train_sampler = None
        self.loss = FasterGSBasisLoss(loss_config=self.LOSS)

    @pre_training_callback(priority=50)
    @torch.no_grad()
    def create_sampler(self, _, dataset: 'BaseDataset') -> None:
        """Creates the sampler."""
        self.train_sampler = DatasetSampler(dataset=dataset.train(), random=True)

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Sets up the model."""
        dataset.train()
        camera_centers = torch.stack([view.position for view in dataset])
        radius = (1.1 * torch.max(torch.linalg.norm(camera_centers - torch.mean(camera_centers, dim=0), dim=1))).item()
        Logger.log_info(f'training cameras extent: {radius:.2f}')

        if dataset.point_cloud is not None:
            point_cloud = dataset.point_cloud
        else:
            n_random_points = 100_000
            min_bounds, size = dataset.bounding_box.min, dataset.bounding_box.size
            positions = torch.rand((n_random_points, 3), dtype=torch.float32, device=size.device) * size + min_bounds
            point_cloud = BasicPointCloud(positions)
        self.model.gaussians.initialize_from_point_cloud(point_cloud)
        self.model.gaussians.training_setup(self, radius)
        self.model.gaussians.reset_densification_info()

    @training_callback(priority=110, start_iteration=1000, iteration_stride=1000)
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase the number of used SH coefficients up to a maximum degree."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, _) -> None:
        """Apply densification."""
        # self.model.gaussians.densify_and_prune(self.DENSIFICATION_GRAD_THRESHOLD, 0.005, iteration > self.OPACITY_RESET_INTERVAL)
        self.model.gaussians.adaptive_density_control(self.DENSIFICATION_GRAD_THRESHOLD, 0.005, iteration > self.OPACITY_RESET_INTERVAL)
        if iteration < self.DENSIFICATION_END_ITERATION:
            self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()

    @training_callback(priority=90, start_iteration='OPACITY_RESET_INTERVAL', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='OPACITY_RESET_INTERVAL')
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Reset opacities."""
        self.model.gaussians.reset_opacities()

    @training_callback(priority=90, start_iteration='EXTRA_OPACITY_RESET_ITERATION', end_iteration='EXTRA_OPACITY_RESET_ITERATION')
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Reset opacities one additional time when using a white background."""
        # original implementation only supports black or white background, this is an attempt to make it work with any color
        if dataset.default_camera.background_color.sum() != 0.0:
            Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Performs a training step without actually doing the optimizer step."""
        # init modes
        self.model.train()
        dataset.train()
        self.loss.train()
        # update learning rate
        self.model.gaussians.update_learning_rate(iteration + 1)
        # get random view
        view = self.train_sampler.get(dataset=dataset)['view']
        # render
        image = self.renderer.render_image_training(
            view=view,
            update_densification_info=iteration < self.DENSIFICATION_END_ITERATION,
        )
        # calculate loss
        # compose gt with background color if needed  # FIXME: integrate into data model
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, view.camera.background_color)
        loss = self.loss(image, rgb_gt)
        # backward
        loss.backward()
        # optimizer step
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()

    @training_callback(active='WANDB.ACTIVATE', priority=10, iteration_stride='WANDB.INTERVAL')
    @torch.no_grad()
    def log_wandb(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Adds Gaussian count to default Weights & Biases logging."""
        Framework.wandb.log({
            '#Gaussians': self.model.gaussians.means.shape[0]
        }, step=iteration)
        # default logging
        super().log_wandb(iteration, dataset)

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        """Clean up after training."""
        n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )
