#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="collaborative-neural-painting",
    version="0.0.1",
    description="",
    author="Nicola Dall'Asen",
    author_email="nicola.dallasen@gmail.com",
    url="https://github.com/fodark/collaborative-neural-painting",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
