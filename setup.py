import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="fastax",
    version="0.1.2",
    description="A Jax based neural network library for research",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/joaogui1/fastax",
    author="João Guilherme Madeira Araújo",
    author_email="joaoguilhermearujo@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["jax", "jaxlib", "numpy"],
)