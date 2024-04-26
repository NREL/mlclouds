A machine learning approach to predicting missing cloud properties in the National Solar Radiation Database (NSRDB)
====================================================================================================================

The National Solar Radiation Database (NSRDB) is NREL’s flagship solar data resource. With over 20 years of high-resolution surface irradiance
data covering most of the western hemisphere, the NSRDB is a crucial public data asset. A fundamental input to accurate surface irradiance in the
NSRDB is high quality cloud property data. Cloud properties are used in radiative transfer calculations and are sourced from satellite imagery.
Improving the accuracy of cloud property inputs is a tractable method for improving the accuracy of the irradiance data in the NSRDB. For example,
in July of 2018, an average location in the Continental United States is missing cloud property data for nearly one quarter of all daylight cloudy timesteps.
This project aims to improve the cloud data inputs to the NSRDB by using machine learning techniques to exploit the NSRDB’s massive data resources.
More accurate cloud property input data will yield more accurate surface irradiance data in the NSRDB, providing direct benefit to researchers at NREL
and to public data users everywhere.

Installation
------------
It is recommended that you first follow the `install instructions for the NSRDB <https://github.com/NREL/nsrdb>`_.
Then run `pip install -e .` from the mlclouds directory containing setup.py.
If you are a developer, also run `pre-commit install` in the same directory.