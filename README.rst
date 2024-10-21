####################
Welcome to MLClouds!
####################
|Docs| |Tests| |Linter| |PyPi| |PythonV| |Codecov| |Zenodo|

.. |Docs| image:: https://github.com/NREL/mlclouds/workflows/Documentation/badge.svg
    :target: https://nrel.github.io/mlclouds/

.. |Tests| image:: https://github.com/NREL/mlclouds/workflows/Pytests/badge.svg
    :target: https://github.com/NREL/mlclouds/actions?query=workflow%3A%22Pytests%22

.. |Linter| image:: https://github.com/NREL/mlclouds/workflows/Lint%20Code%20Base/badge.svg
    :target: https://github.com/NREL/mlclouds/actions?query=workflow%3A%22Lint+Code+Base%22

.. |PyPi| image:: https://img.shields.io/pypi/pyversions/NREL-mlclouds.svg
    :target: https://pypi.org/project/NREL-mlclouds/

.. |PythonV| image:: https://badge.fury.io/py/NREL-mlclouds.svg
    :target: https://badge.fury.io/py/NREL-mlclouds

.. |Codecov| image:: https://codecov.io/gh/nrel/mlclouds/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/nrel/mlclouds

.. |Zenodo| image:: https://zenodo.org/badge/340209614.svg
    :target: https://zenodo.org/badge/latestdoi/340209614

.. inclusion-intro


A machine learning approach to predicting missing cloud properties in the National Solar Radiation Database (NSRDB)
-------------------------------------------------------------------------------------------------------------------

The National Solar Radiation Database (NSRDB) is NREL's flagship solar data resource. With over 20 years of high-resolution surface irradiance
data covering most of the western hemisphere, the NSRDB is a crucial public data asset. A fundamental input to accurate surface irradiance in the
NSRDB is high quality cloud property data. Cloud properties are used in radiative transfer calculations and are sourced from satellite imagery.
Improving the accuracy of cloud property inputs is a tractable method for improving the accuracy of the irradiance data in the NSRDB. For example,
in July of 2018, an average location in the Continental United States is missing cloud property data for nearly one quarter of all daylight cloudy timesteps.
This project aims to improve the cloud data inputs to the NSRDB by using machine learning techniques to exploit the NSRDB's massive data resources.
More accurate cloud property input data will yield more accurate surface irradiance data in the NSRDB, providing direct benefit to researchers at NREL
and to public data users everywhere.

Installation
============

It is recommended that you first follow the `install instructions for the NSRDB <https://github.com/NREL/nsrdb>`_.
Then run ``pip install -e .`` from the mlclouds directory containing ``setup.py``.
If you are a developer, also run ``pre-commit install`` in the same directory.


Acknowledgments
===============

This work (SWR-23-77) was authored by the National Renewable Energy Laboratory,
operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of
Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE
Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research
(ASCR) program, the DOE Solar Energy Technologies Office (SETO), the DOE Wind
Energy Technologies Office (WETO), the United States Agency for International
Development (USAID), and the Laboratory Directed Research and Development
(LDRD) program at the National Renewable Energy Laboratory. The research was
performed using computational resources sponsored by the Department of Energy's
Office of Energy Efficiency and Renewable Energy and located at the National
Renewable Energy Laboratory. The views expressed in the article do not
necessarily represent the views of the DOE or the U.S. Government. The U.S.
Government retains and the publisher, by accepting the article for publication,
acknowledges that the U.S. Government retains a nonexclusive, paid-up,
irrevocable, worldwide license to publish or reproduce the published form of
this work, or allow others to do so, for U.S. Government purposes.