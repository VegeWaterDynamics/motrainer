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

`MOTrainer` requires `h5py` for installation. Usually your Python dependency manager should hamdle the installation of `h5py` automatically. However, the following error may occur when installing `h5py`:

```output
ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based project
```

On Ubuntu system, this can be resolved by installing the `libhdf5-dev` package:

```bash
sudo apt-get install libhdf5-dev
```

On MacOS, h5py can be installed via `Homebrew`, `Macports`, or `Fink`. See the [installation guide of h5py](https://docs.h5py.org/en/stable/build.html#os-specific-package-managers)

According to the [installation guide of h5py](https://docs.h5py.org/en/stable/build.html#os-specific-package-managers) there is no Windows-specific package managers include `h5py`. Therefore, it is recommended to install `h5py` on Windows via Python Distributions such as`conda`.

```bash
conda install h5py
```
