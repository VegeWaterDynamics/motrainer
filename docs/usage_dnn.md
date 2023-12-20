# Parallel DNN Training with JackknifeGPI

The [data split section](https://vegewaterdynamics.github.io/motrainer/usage_split/) explains how to partition a dataset. After partitioning, the dataset is divided into two data bags: `train_bags` and `test_bags`.

This section demonstrates how to perform distributed DNN training using Tensorflow, with [Jackknife resampling](https://en.wikipedia.org/wiki/Jackknife_resampling) as the cross-validation method. In this approach, each year is iteratively left out as the cross-validation data. The best model is selected based on the lowest Root Mean Square Error (RMSE).

A more comprehensive example can be found in this [Example Notebook](https://vegewaterdynamics.github.io/motrainer/notebooks/example_dnn/).


## Training a Single Grid Point

To train a DNN for a single grid cell, you can use the `JackknifeGPI` object:

```python
from motrainer.jackknife import JackknifeGPI

# Intiate a Jackknife GPI from one gridcell
df = train_bags.take(1)
gpi_data = df.compute()
gpi = JackknifeGPI(gpi_data, outpath='./results')

# Perform training and export
results = gpi.train()
gpi.export_best()
```

The training results will be exported to the `./results `path.

## Training Multiple Grid Points

To train multiple grid points in parallel, you can define a training function as follows:

```python
def training_func(gpi_num, df):
    gpi_data = df.compute()

    gpi = JackknifeGPI(gpi_data,
                       outpath=f"results/gpi{gpi_num}")

    gpi.train()
    gpi.export_best()
```

Then, map the training function to each grid cell:

```python
from dask.distributed import Client, wait

# Use client to parallelize the loop across workers
client = Client()
futures = [
    client.submit(training_func, gpi_num, df) for gpi_num, df in enumerate(train_bags)
]

# Wait for all computations to finish
wait(futures)

# Get the results
results = client.gather(futures)
```

The above examples uses a local threaded Dask scheduler to parallelize the tasks. When executing training on an HPC system, we recommend using [Dask SLURM cluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html) for the distributed training. For more information on different Dask clusters, please check the [Dask Documentation](https://docs.dask.org/en/stable/deploying.html).

You can also directly submit training jobs as SLURM jobs, instead of using Dask SLURM cluster. You can find the example of using SLURM [here](../example/slurm_examples/readme.md).