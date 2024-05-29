# MOTrainer: Measurement Operator Trainer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7540442.svg)](https://doi.org/10.5281/zenodo.7540442)
[![PyPI](https://img.shields.io/pypi/v/motrainer.svg?colorB=blue)](https://pypi.python.org/project/motrainer/)
[![Build](https://github.com/VegeWaterDynamics/motrainer/actions/workflows/build.yml/badge.svg)](https://github.com/VegeWaterDynamics/motrainer/actions/workflows/build.yml)
[![Ruff](https://github.com/VegeWaterDynamics/motrainer/actions/workflows/lint.yml/badge.svg)](https://github.com/VegeWaterDynamics/motrainer/actions/workflows/lint.yml)

Measurement Operator Trainer is a Python package training measurement operators (MO) for data assimilations purposes. It is specifically designed for the applications where one needs to split large spatio-temporal data into independent partitions, and then train separate ML models for each partition.

Please refer to the [MOtrainer documentation](https://vegewaterdynamics.github.io/motrainer/) for more details.

## Installation

Python version `>=3.10` is required to install MOTrainer.

MOTrainer can be installed from PyPI:

```sh
pip install motrainer
```

We suggest using [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#fresh-install-recommended) to create an isolated environment for the installation to avoid conflicts.

For more details and trouble shooting of the installation process, please refer to the [installation guide](https://vegewaterdynamics.github.io/motrainer/setup/) for more details.

## License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).