#!/usr/bin/env python

"""polarizability_model

Simulate and invert
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

with open("README.md") as f:
    LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="polarizability_model",
    version="0.0.1",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "numpy>=1.7",
        "scipy>=1.0.0",
        "SimPEG"
    ],
    author="Lindsey Heagy",
    author_email="lheagy@eoas.ubc.ca",
    description="polarizability_model",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="unexploded ordnance, magnetics, electromagnetics",
    url="https://github.com/lheagy/uxo-feature-design",
    download_url="https://github.com/lheagy/uxo-feature-design",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
)
