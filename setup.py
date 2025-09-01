from setuptools import setup, find_packages


setup(
    name="direct3d_s2",
    version="1.0.0",
    description="Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "cython",
        "trimesh",
        "diffusers",
        "triton",
    ],
)