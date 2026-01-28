from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

HERE = Path(__file__).parent.resolve()

setup(
    name="feature_dataset",
    version="0.1.0",
    packages=["feature_dataset"],
    package_dir={"feature_dataset": str(HERE)},
    python_requires=">=3.10",
    ext_modules=[
        CppExtension(
            name="feature_dataset._C",
            sources=[str(HERE / "csrc/feature_dataset.cpp")],
            libraries=["zstd"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fvisibility=hidden", "-pthread"],
            },
            extra_link_args=["-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
