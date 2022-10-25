################################################################################
ml_lsmodel_ascat
################################################################################

ml_lsmodel_ascat is a Machine Learning package written in Python. 
It trains surrogate model to connect soil and vegetation states/parameters to microwave observations.

Installation
------------

First, please clone this repository to prepare for installation.

.. code-block:: console

  git clone https://github.com/VegeWaterDynamics/ml_lsmodel_ascat.git

We recommend to install ``ml_lsmodel_ascat`` via ``mamba``. It can either be installed independently, or under 
the ``base`` environment of ``conda``.
Please check the `installation guide <https://mamba.readthedocs.io/en/latest/installation.html>`_ to install ``mamba``.


After you succesfully installed ``mamba``, you can install the environment from the ``environment.yml`` file:  

.. code-block:: console

  mamba env create -f environment.yml
  
A new environment with the name ``vegetation`` will be created. You can activate it by:

.. code-block:: console

  mamba activate env_ml_lsmodel_ascat

To make sure everything works, you can run:

.. code-block:: console

  pytest tests/

Documentation
*************

.. _README:

Currently the package is not publicly released. Please refer to the source code for documentation.

Contributing
************

If you want to contribute to the development of ml_lsmodel_ascat,
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
