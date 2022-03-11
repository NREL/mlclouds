"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "md", format="md")

with open("requirements.txt") as f:
    install_requires = f.readlines()

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
    package_data={'mlclouds':
                  ['model/production_model/outputs/mlclouds_model.pkl',
                   'model/production_model/outputs/mlclouds_model.json']},
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="mlclouds",
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    install_requires=install_requires,
    )
