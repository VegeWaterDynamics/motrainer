################################################################################
ml_lsmodel_ascat
################################################################################

ml_lsmodel_ascat is a Machine Learning package written in Python. 
It trains surrogate model to connect soil and vegetation states/parameters to microwave observations.

Installation
------------

We recommend to install ``ml_lsmodel_ascat`` under a ``conda`` environment. You can do this via ``miniconda``.
Please check the `installation tutorial <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_ to install ``miniconda``.

After you succesfully installed ``miniconda``, you can create and activate a new environment:  

.. code-block:: console

  conda create -n env_ml_lsmodel_ascat
  conda activate env_ml_lsmodel_ascat

There is also a `cheat sheet <https://conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf>`_ for more commands of conda environment management. 

Now we are going to install ``ml_lsmodel_ascat`` under our newly created environment ``env_ml_lsmodel_ascat``.
First you need to make sure you are in a desired working directory.
Then the installation can be done simpy by:

.. code-block:: console

  git clone https://github.com/VegeWaterDynamics/ml_lsmodel_ascat.git
  cd ml_lsmodel_ascat
  pip install .

By this we clone the Github repository to local and install it with all dependencies. 
Well, all dependencies but one: ``Cartopy``, which is used for managing geospatial projection system for visulization.
Unfortuantely ``Cartopy`` has a very poor ``pip`` support. We can manually install it via ``conda``:

.. code-block:: console

  conda install -c conda-forge cartopy

The installation is complete.
Alternatively, to check if everything is working, you can run unittests:


.. code-block:: console

  python3 setup.py test


Documentation
*************

.. _README:

Currently the package is not publicly released. Plz refer to the source code for documentation.

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
