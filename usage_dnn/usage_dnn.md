# Parallel DNN Training with JackknifeGPI

## Train one grid point
```python
df = train_bags.take(1)
gpi_data = df.compute()
gpi_data.dropna(inplace=True)
```

```python
from motrainer.jackknife import JackknifeGPI

# Intiate a Jackknife GPI with initial settings
gpi = JackknifeGPI(gpi_data, outpath='./results')

# Perform training and export
results = gpi.train()
gpi.export_best()
```

## Train multiple grid point
```python
def training_func(gpi_num, df):
    # remove NA data
    gpi_data = df.compute()
    gpi_data.dropna(inplace=True)

    gpi = JackknifeGPI(gpi_data,
                       outpath=f"results/gpi{gpi_num}")

    gpi.train()
    gpi.export_best()
```

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