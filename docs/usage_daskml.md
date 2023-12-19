# Parallel ML Model Optimization with Dask-ml

In the [data split section](https://vegewaterdynamics.github.io/motrainer/usage_split/), we discussed how to partition a dataset. After the partitioning process, the dataset is divided into two data bags: `train_bags` and `test_bags`.

Following dataset partitioning, you can execute parallel machine learning training using [`dask-ml`](https://ml.dask.org/). Dask-ml leverages Dask alongside popular machine learning libraries like Scikit-Learn to facilitate distributed machine learning tasks. This document provides a brief overview of its usage. For a more comprehensive example, refer to this [Jupyter Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_daskml/).


## Extract training data

With `dask-ml`, the input and output data need be converted to a DataFrame. You can map this conversion to each partition as follows:

```python
def to_dataframe(ds):
    return ds.to_dask_dataframe()

def chunk(ds, chunks):
    return ds.chunk(chunks)

train_bags = train_test_bags.pluck(0).map(chunk, {"space": 500}).map(to_dataframe)
```

## Configuring the Search Grid

You can set up distributed machine learning training jobs using [`dask-ml.model_selection`](https://ml.dask.org/modules/api.html#module-dask_ml.model_selection). For instance, you can perform an exhaustive search over specified parameter values for an estimator using `GridSearchCV`:

```python
import dask_ml.model_selection as dcv
from sklearn import svm, datasets

parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
svc = svm.SVC()
search = dcv.GridSearchCV(svc, parameters)
```

## Executing Grid Search in Parallel

After setting up the search object, you can map the `fit` function to each partition as follows:

```python
def optimize(df, grid_search, input_list, output_list):
    """Customized Optimization Function
    """
    grid_result = grid_search.fit(df[input_list], df[output_list])
    return grid_result

input_list = ["STATE1", "STATE2", "STATE3", "STATE4", "STATE5"]
output_list = ["slop"]

# When memory allows, peresist data in the workers
# When train_bags can is lazilly loading from file system, this can avoid redundant data loading
train_bags = train_bags.persist()

optimazed_estimators = train_bags.map(
    optimize, search, input_list, output_list
)
```

In this way, the machine learning training task is parallelized on two levels:

1. The training jobs of all partitions of the dataset are parallelized;
2. The parameter searching within each partition is parallelized.

By default, Dask uses a local threaded scheduler to parallelize the tasks. Alternatively, other types of clusters can be set up if the training job is running on other infrastructures. The usage of different clusters will not influence the syntax of data split and training jobs. For more information on different Dask clusters, please check the [Dask Documentation](https://docs.dask.org/en/stable/deploying.html).