from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).parent.resolve()

setup(
    name="dataset",
    version="0.1.0",
    packages=["dataset"],
    package_dir={"dataset": str(HERE)},
    python_requires=">=3.10",
    ext_modules=[
        CppExtension(
            name="dataset._C",
            sources=[
                str(HERE / "csrc/dataset.cpp"),
                str(HERE / "csrc/bin_dataset.cpp"),
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-fvisibility=hidden",
                    "-pthread",
                    "-march=native",
                ],
            },
            extra_link_args=["-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
