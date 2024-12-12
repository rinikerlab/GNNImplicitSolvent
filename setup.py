from setuptools import setup, find_packages

setup(
    name="GNNImplicitSolvent",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"GNNImplicitSolvent": "."},
    package_data={
        "Simulation": [
            "solvents.yml",
        ],
        "MachineLearning": [
            "trained_models/GNN.pt",
            "trained_models/GNN_WATER_ONLY.pt",
        ],
    },
    author="Paul Katzberger",
    author_email="kpaul@ethz.ch",
    description="A package for GNN-based implicit solvent",
    url="https://github.com/rinikerlab/GNNImplicitSolvent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
