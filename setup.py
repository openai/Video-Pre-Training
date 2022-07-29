

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name="vpt",
    version="1.0",
    description="Video Pre Training for MineRL",
    long_description=long_description,
    packages=["vpt"],
    install_requires=requirements,
)