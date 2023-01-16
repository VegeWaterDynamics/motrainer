################################################################################
MOTrainer: Measurement Operator Trainer
################################################################################

.. image:: https://zenodo.org/badge/299843809.svg
   :target: https://zenodo.org/badge/latestdoi/299843809

Measurement Operator Trainer is a Python package training measurement operators 
(MO) for data assimilations purposes. At present, the MOTrainer trains Deep Neuron
Networks as measurement operators for Earth Observation (EO) application. The 
trained predicts sensor measurement based on certain physical model states.

MOTrainer assumes the availability of the training dataset, i.e. input (model 
status) and output (sensor data).

Installation
------------

First, please clone this repository to prepare for installation.

.. code-block:: console

  git clone https://github.com/VegeWaterDynamics/motrainer.git

We recommend installing ``MOtrainer`` under a blank `conda environment 
<https://docs.conda.io/en/latest/>`_. After activating the conda environment,
you can install ``MOTrainer`` from the source:

.. code-block:: console
  
  cd motrainer
  pip install .


Alternatively, you can also install ``motrainer`` via ``conda``. 

.. code-block:: console

  conda env create -f environment.yml


Usage examples
*************

The usage example of MOTrainer can be found `here <example/demo_jackknife.py>`_

Alternatively, the execution of MOTrainer can be scaled up to an HPC system by using
SLURM. Please refer to the example `here <example/demo_slurm/readme.md>`_.


Contributing
************

If you want to contribute to the development of MOtrainer,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2021, 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



Credits
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
