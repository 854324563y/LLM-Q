from setuptools import setup
from torch.utils import cpp_extension
import torch
from pathlib import Path

CUTLASS_PATH = "/workspace/volume/yangzhe/cutlass"  # 请替换为您的CUTLASS安装路径
CUDA_PATH = cpp_extension.CUDA_HOME
TORCH_PATH = Path(torch.__file__).parent

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

setup(
    name='int_kernel',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='int_kernel._CUDA',
            sources=[
                'int_kernel/kernels/bindings.cpp',
                # 'int_kernel/kernels/int_kernel.cu',
                'int_kernel/kernels/s8t_s8n_f16t_kernel.cu',
                'int_kernel/kernels/s4t_s4n_f16t_kernel.cu',
                'int_kernel/kernels/s8t_s4n_f16t_kernel.cu',
            ],
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    f'-I{CUDA_PATH}/include',
                    f'-I{TORCH_PATH}/include',
                    f'-I{TORCH_PATH}/include/torch/csrc/api/include',
                    f'-I{CUTLASS_PATH}/include',
                    f'-I{CUTLASS_PATH}/tools/util/include',
                    '-std=c++17',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__'
                ],
                'nvcc': [
                    '-O3',
                    f'-I{CUDA_PATH}/include',
                    f'-I{TORCH_PATH}/include',
                    f'-I{TORCH_PATH}/include/torch/csrc/api/include',
                    f'-I{CUTLASS_PATH}/include',
                    f'-I{CUTLASS_PATH}/tools/util/include',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    #'--extended-lambda',
                    #'--expt-relaxed-constexpr',
                    #'--use_fast_math',
                    #f'-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}',
                    f'-DCUDA_ARCH={cuda_arch}',
                ]
            },
            include_dirs=[
                'int_kernel/kernels/include',  # 项目头文件目录
                f'{CUDA_PATH}/include',
                f'{TORCH_PATH}/include',
                f'{TORCH_PATH}/include/torch/csrc/api/include',
                f'{CUTLASS_PATH}/include',
                f'{CUTLASS_PATH}/tools/util/include'
            ],
            library_dirs=[
                f'{CUDA_PATH}/lib64',
                f'{TORCH_PATH}/lib'
            ],
            libraries=['cudart', 'c10', 'torch', 'torch_cpu', 'torch_python']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=False)
    }) 