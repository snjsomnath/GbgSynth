"""Setup configuration for the GbgSynth library."""

import re
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Single source of truth: read version from gbgsynth/__init__.py
with open("gbgsynth/__init__.py", "r", encoding="utf-8") as fh:
    _version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', fh.read(), re.M)
    if not _version_match:
        raise RuntimeError("Unable to find __version__ in gbgsynth/__init__.py")
    _version = _version_match.group(1)

setup(
    name="gbgsynth",
    version=_version,
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
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.10.0",
        "geopandas>=0.12.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    package_data={
        "gbgsynth": ["config/*.json"],
    },
    include_package_data=True,
)
