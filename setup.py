#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

from evnrg import __version__
# from requirements import requirements, dependency_links

requirements = [
    'numba>=0.42.0',
    'numpy>=1.16.0',
    'pandas>=0.24.0',
    'dask',
    'distributed',
    'appdirs',
    'apache-libcloud',
    'fastparquet',
    'pycrypto'
    'seaborn>=0.9.0'
]

dependency_links = []

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Alex Campbell",
    author_email='amcampbell@ucdavis.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="EVNRG is an EV electrical demand simulation package that takes in trip data and turns it into useful energy data given a set of assumptions.",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='evnrg',
    name='evnrg',
    packages=find_packages(include=['evnrg']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/accurrently/evnrg',
    version=__version__,
    zip_safe=False,
    dependency_links=dependency_links
)
