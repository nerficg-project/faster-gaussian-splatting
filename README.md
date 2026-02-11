# Faster Gaussian Splatting
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=PyTorch&logoColor=white)&nbsp;
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=NVIDIA&logoColor=white)&nbsp;
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)

<img src="resources/teaser.gif" height="342"/>

__Progress accelerates when feedback cycles shrink.__  
Ever wondered how much faster you could iterate in your [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) research by unifying the most impactful advances from its many follow-up works?  
This repository presents our answer: the official implementation of “_Faster-GS: Analyzing and Improving Gaussian Splatting Optimization_”, an updated baseline for 3DGS optimization.

> __Faster-GS: Analyzing and Improving Gaussian Splatting Optimization__  
> [Florian Hahlbohm](https://fhahlbohm.github.io), [Linus Franke](https://lfranke.github.io/), [Martin Eisemann](https://graphics.tu-bs.de/people/eisemann), [Marcus Magnor](https://graphics.tu-bs.de/people/magnor)  
 _arXiv, February 2026_  
> __[Project page](https://fhahlbohm.github.io/faster-gaussian-splatting)&nbsp;| [Paper](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf)&nbsp;| [BibTeX](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.bib)__


## Overview

Faster-GS is a highly-optimized yet readable and extensible 3DGS implementation.
Compared to other research codebases that are currently available, it is roughly 2-5x faster and uses less VRAM.
We achieve this by combining ideas from multiple 3DGS follow-up works with our own improvements. If you are interested in the details, we recommend you read our paper.

This repository has different branches that provide variants of Faster-GS. While we intentionally kept these variants separate to improve readability and modularity, there are no fundamental limitations with combining their respective features.

The `main` branch provides our fast 3DGS implementation with tons of additional features. We hope that their joint availability within an optimized framework will accelerate future research.

- Anti-aliasing improvements based on [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/)
- MCMC densification based on [3DGS-MCMC](https://ubc-vision.github.io/3dgs-mcmc/)
- Different Gaussian truncation modes enabled by a revised opacity interpretation within the rasterizer
- Improved random initialization through visibility and/or mask-based carving
- Random background color training to reduce false transparency
- (coming soon) Even faster training through informed pruning from [Speedy-Splat](https://speedysplat.github.io/)
- (coming soon) Handling of photometric variations in input images through [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/)
- (coming soon) Training on distorted images using [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/)

On the `FasterGSFused` branch, we went full nerd-mode and fused the backward pass and optimizer. It provides an additional speedup over our default implementation and also uses less VRAM. And the best part is that the code is still pretty clean. Great if you need every bit of performance or just like to train extra fast.

The `FasterGS4D` branch provides an extension of our implementation to dynamic scene reconstruction with [4D Gaussians](https://fudan-zvg.github.io/4d-gaussian-splatting/). Compared to the original implementation by Yang et al., ours trains significantly faster and uses less VRAM without sacrificing quality.

The `FasterGSTestbed` branch contains the implementation we used to develop, integrate, and evaluate all improvements we discuss in our paper. It allows you to individually enable/disable the components we evaluate in the ablation study of our paper. This branch is a great starting point if, for example, you can not afford the extra VRAM required by the improved backward pass and want to develop a VRAM-optimized implementation.

On the `FasterGSBasis` branch, we provide our basis implementation described in the appendix of our paper. All other variants are built on top of this basis. We used it to develop and test multiple minor algorithmic and cleanup changes to ensure that they do not affect quality.


## Installation

Our implementation is provided as an extension to [NeRFICG](https://github.com/nerficg-project), a radiance field and view synthesis framework actively maintained and developed by our research group.

__Easy integration into existing codebases:__  
We understand that for compatibility with previous or current projects you might not be able to immediately switch from your current 3DGS framework.
Therefore, we tried to make it as easy as possible to integrate the most impactful improvements into exisiting codebases. Our CUDA backend can be installed into an exisiting environment as a PyTorch extension through the following command:
```shell
pip install git+https://github.com/nerficg-project/faster-gaussian-splatting/#subdirectory=FasterGSCudaBackend --no-build-isolation
```
Integrating the key features of this extension into the official 3DGS codebase (`graphdeco-inria/gaussian-splatting`) requires only minimal code changes.  
A reference implementation is available [here](https://github.com/fhahlbohm/gaussian-splatting).

__That being said, we encourage you to give the native implementation a shot as it provides the full performance improvements as well as a ton of additional features. We also tried to make the code as readable and extensible as possible by optimizing and refactoring the PyTorch and CUDA code of all integrated methods.__


### Requirements

- An NVIDIA GPU
- Linux (preferred) or Windows
- A recent CUDA SDK ([CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) recommended) and a compatible C++ compiler
- [Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed


### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Please follow the instructions in its README to set up a compatible Conda environment.

Now add Faster-GS as an additional method by cloning this repository into `src/Methods/FasterGS`:
```shell
# HTTPS
git clone https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
```
or
```shell
# SSH
git clone git@github.com:nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
```

Next, install all method-specific dependencies and CUDA extensions using:
```shell
python ./scripts/install.py -m FasterGS
```

_Note: The framework determines on-the-fly what extra modules need to be installed. Sometimes this causes unnecessary errors/warnings that can interrupt the installation process. In this case, first try to rerun the command before investigating the error in detail._


### Variants

Installing one of our Faster-GS variants (e.g., `FasterGS4D`) from their respective branch can be achieved by
```shell
# HTTPS
git clone --single-branch -b FasterGS4D https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS4D
```
or
```shell
# SSH
git clone --single-branch -b FasterGS4D git@github.com:nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS4D
```
and then
```shell
python ./scripts/install.py -m FasterGS4D
```


## Usage

Coming soon ...

### Training and Inference

The Faster-GS method is fully compatible with the NeRFICG scripts in the `scripts/` directory.
This includes config file generation via `create_config.py`,
training via `train.py`,
inference and performance benchmarking via `inference.py`,
metric calculation via `generate_tables.py`,
and live rendering via `gui.py`.
For development, we also use the `benchmark.py` script that allows easy evaluation of a given configuration file for all scenes in a dataset.

For detailed instructions, please refer to the [NeRFICG repository](https://github.com/nerficg-project/nerficg).


## Acknowledgements

This work was partially funded by the DFG projects
[Real-Action VR](https://graphics.tu-bs.de/projects/real-action-vr) (ID 523421583) and
[Increasing Realism of Omnidirectional Videos in Virtual Reality](https://graphics.tu-bs.de/projects/increasing-perceived-realism-in-omnidirectional-visual-media) (ID 491805996).  
Linus Franke was supported by the ERC Advanced Grant [NERPHYS](https://project.inria.fr/nerphys) (ID 101141721).

We thank the authors of the following works, whose ideas and open-source implementations form the foundation of this project:
- Kerbl et al. [_3D Gaussian Splatting for Real-Time Radiance Field Rendering._](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) SIGGRAPH 2023.
- Radl et al. [_StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering._](https://r4dl.github.io/StopThePop/) SIGGRAPH 2024.
- Mallick et al. [_Taming 3DGS: High-Quality Radiance Fields with Limited Resources._](https://humansensinglab.github.io/taming-3dgs/) SIGGRAPH Asia 2024.
- Schütz et al. [_Splatshop: Efficiently Editing Large Gaussian Splat Models._](https://publications.graphics.tudelft.nl/papers/822) HPG 2025.
- Yang et al. [_Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting._](https://fudan-zvg.github.io/4d-gaussian-splatting/) ICLR 2024.
- Yu et al. [_Mip-Splatting: Alias-free 3D Gaussian Splatting._](https://niujinshuchong.github.io/mip-splatting/) CVPR 2024.
- Kheradmand et al. [_3D Gaussian Splatting as Markov Chain Monte Carlo._](https://ubc-vision.github.io/3dgs-mcmc/) NeurIPS 2024.

We also thank [Janusch Patas (aka. MrNeRF)](https://github.com/MrNeRF) for partially inspiring this project with his open-source bounty for [LichtFeld Studio](https://lichtfeld.io/), where an early version of our work was the [winning submission](https://github.com/MrNeRF/LichtFeld-Studio/pull/245) and has since become one of its core components.


## License and Citation

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

If you use this project in your research, please cite our paper:

```bibtex
@misc{hahlbohm2026fastergs,
  title         = {Faster-GS: Analyzing and Improving Gaussian Splatting Optimization},
  author        = {Florian Hahlbohm and Linus Franke and Martin Eisemann and Marcus Magnor},
  year          = {2026},
  eprint        = {2602.09999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2602.09999},
}
```
