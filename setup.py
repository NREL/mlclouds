"""
setup.py
"""
import os
from codecs import open

from setuptools import find_packages, setup

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "rst", format="rst")

with open("requirements.txt") as f:
    install_requires = f.readlines()

with open(os.path.join(here, "mlclouds", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")

setup(
    name="NREL-mlclouds",
    version=version,
    description="Machines Learning Clouds",
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL",
    packages=find_packages(),
    package_dir={"mlclouds": "mlclouds"},
    package_data={'mlclouds':
                  ['model/production_model/outputs/mlclouds_model.pkl',
                   'model/production_model/outputs/mlclouds_model.json']},
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="mlclouds",
    python_requires='>=3.9',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="tests",
    install_requires=install_requires,
    )
