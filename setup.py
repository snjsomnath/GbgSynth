"""Setup configuration for the GbgSynth library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gbgsynth",
    version="0.3.0",
    author="Sanjay Somanath",
    author_email="sanjay.somanath@chalmers.se",
    description="Synthetic population generator for Gothenburg using PxWeb API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snjsomnath/gbgsynth",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "geopandas>=0.12.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    package_data={
        "gbgsynth": ["config/*.json"],
    },
    include_package_data=True,
)
