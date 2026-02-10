import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'A refactored CUDA implementation of the 3DGS rasterizer.'

ENABLE_FASTMATH = True  # set to False to disable fast math optimizations (e.g., for debugging)
ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

# gather source files
sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
for module in cuda_modules:
    sources += glob(str(extension_root / module / 'src' / '**'/ '*.cpp'), recursive=True)
    sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

# gather include directories
include_dirs = [str(extension_root / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

# set up compiler flags
cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_FASTMATH:
    cxx_flags.append('-O3')
    nvcc_flags.append('-O3')
    nvcc_flags.append('-use_fast_math')
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

# define the CUDA extension
extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
