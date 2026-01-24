from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).parent.resolve()

setup(
    name='onehot_sparse_linear',
    version='0.1.0',
    packages=['sparse_linear'],
    package_dir={'sparse_linear': str(HERE)},
    python_requires='>=3.10',
    ext_modules=[
        CUDAExtension(
            name='sparse_linear._C',
            sources=[
                str(HERE / 'csrc/onehot_sparse_linear.cpp'),
                str(HERE / 'csrc/onehot_sparse_linear_cuda.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
