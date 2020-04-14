"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
import sys

py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("MLClouds is only compatible with python 3.7!")
elif py_version.minor != 7:
    raise RuntimeError("MLClouds is only compatible with python 3.7!")

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "md", format="md")

setup(
    name="mlclouds",
    version="0.0.0",
    description="Machines Learning Clouds",
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.nrel.gov/PXS/mlclouds",
    packages=find_packages(),
    package_dir={"mlclouds": "mlclouds"},
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="mlclouds",
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    test_suite="tests",
    install_requires=["nsrdb>=3.0",
                      "numpy>=1.16",
                      "pandas>=0.25",
                      "scipy>=1.3",
                      "matplotlib>=3.1",
                      "pytest>=5.2",
                      "scikit-learn>=0.22",
                      "tensorflow>=2.0",
                      "ipython",
                      "notebook",
                      "psutil",
                      "pre-commit",
                      "flake8",
                      "pylint",
                      "NREL-reV"],
)
