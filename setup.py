# coding: utf-8
"""
Setup script for numericalderivative
====================================

This script allows to install numericalderivative within the Python environment.

Usage
-----
::

    python setup.py install

"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="numericalderivative",
    keywords=[
        "Numerical Differentiation",
        "First Derivative",
        "Finite Difference",
        "Optimal Step Size",
        "Numerical Analysis",
    ],
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy"],
    description="Numerical differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
    license="LGPL",
    url="https://github.com/mbaudin47/numericalderivative",
    author="Michaël Baudin",
    author_email="michael.baudin@gmail.com",
)
