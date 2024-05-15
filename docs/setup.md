# Installation

Python version `>=3.10` is required to install MOTrainer.

MOTrainer can be installed from PyPI:

```sh
pip install motrainer
```

or from the source:

```sh
git clone git@github.com:VegeWaterDynamics/motrainer.git
cd motrainer
pip install .
```

## Tips

We strongly recommend installing separately from your default Python envrionment. E.g. you can use enviroment manager e.g. [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) to create separate environment.

## Troubleshooting

### Error: Could not build wheels for h5py

The following error may occur when installing `h5py` on Ubuntu system:

```output
ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based project
```

This can be resolved by installing the `libhdf5-dev` package:

```bash
sudo apt-get install libhdf5-dev
```
