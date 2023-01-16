#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

# To update the package version number, edit CITATION.cff
with open('CITATION.cff', 'r') as cff:
    for line in cff:
        if 'version:' in line:
            version = line.replace('version:', '').strip().strip('"')

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split()

setup(
    name='motrainer',
    version=version,
    description="Machine Learning module to train a surrogate module using Land-surface model dtput",
    long_description=readme + '\n\n',
    author="Netherlands eScience Center, TUDelft",
    author_email='team-atlas@esciencecenter.nl',
    url='https://github.com/VegeWaterDynamics/motrainer',
    packages=[
        'motrainer',
    ],
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='motrainer',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=requirements,  
    setup_requires=[
        # dependency for `python setup.py test`
        'pytest-runner',
        # dependencies for `python setup.py build_sphinx`
        'sphinx',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    data_files=[('citation/motrainer', ['CITATION.cff'])]
)
