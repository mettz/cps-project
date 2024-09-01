from distutils.core import setup

from setuptools import find_packages

setup(
    name="cps_project",
    version="0.1.0",
    author=["Daniele Paccusse", "Mattia Guazzaloca"],
    author_email=[
        "daniele.paccusse@studio.unibo.it",
        "mattia.guazzaloca@studio.unibo.it",
    ],
    license="MIT",
    packages=find_packages(),
    description="CPS project course 2023/2024",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "numpy",
        "torch",
        "pytorch3d",
        "opencv-python",
        "skrl==1.1.0",
        "wandb",
    ],
)
