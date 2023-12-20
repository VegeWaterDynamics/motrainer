# MOTrainer

Measurement Operator Trainer (MOTrainer) is a Python package which facilitates parallel training processes of measurement operators (MO) for Data Assimilation (DA) purposes. It specifically address the need of parallely training multiple MOs over independent partitions of large spatio-temporal dataset.

MOTrainer provices functionality to split spatio-temporal datasets for independent training processes. It utilizes [Xarray](https://docs.xarray.dev/en/stable/index.html)'s feature of multi-dimensional labelling to address the spatio-temporal characteres of the input/output datasets. Then [Dask](https://www.dask.org/) is implemented to achive the parallel training jobs.  

MOTrainer supports training simple structured Machine Learning (ML) models which can be trained using [SciKit-Learn](https://scikit-learn.org/stable/index.html) toolkits. It also provide supports on parallel training Training DeepNeuron Networks with [TensorFlow](https://www.tensorflow.org/).