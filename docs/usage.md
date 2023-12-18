# Usage Examples

MOTrainer is specifically designed for the aplications where one needs to split large spatio-temporal data into independent partitions, and then train separate ML models for each partition. In this page there are some example usage of MOTrainer. 

## Data Preparation

We assume the users starts from an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object which contains spatio-temporal data needed for training multiple independent Machine Learning (ML) models.

```python
print(ds)
```

```output
<xarray.Dataset>
Dimensions:    (space: 5, time: 8506)
Coordinates:
    latitude   (space) float64 56.12 46.12 53.38 49.38 44.38
    longitude  (space) float64 11.38 6.625 6.125 12.38 0.625
  * time       (time) datetime64[ns] 2007-01-02 ... 2020-01-01T01:00:00
Dimensions without coordinates: space
Data variables:
    STATE1     (space, time) float64 0.07079 0.05532 0.04846 ... 0.06712 0.05521
    STATE2     (space, time) float64 0.04366 0.0462 0.03821 ... 0.0861 0.05622
    STATE3     (space, time) float64 280.0 270.4 285.5 277.4 ... 287.4 272.1
    STATE4     (space, time) float64 274.8 278.4 280.6 283.7 ... 280.2 281.5
    STATE5     (space, time) float64 280.9 279.7 278.0 278.0 ... 281.2 280.1
    OBS        (space, time) float64 -9.49 -8.494 -9.069 ... -8.071 -8.237
Attributes:
    license:  data license
    source:   data source
```

```python
motrainer.is_splitable(ds)
```

```output
True
```

## Split Spatio-Temporal Dataset for Independent Training Processes

### Split by dimension names
One can use the `dataset_split` to split data into different partions:

```python
import motrainer
bags = motrainer.dataset_split(ds, "space")
print(bags)
```
```output
dask.bag<from_sequence, npartitions=5>
```

The split results can be checked by:
```python
print(bags.take(1))
```
```output
(<xarray.Dataset>
 Dimensions:    (space: 1, time: 8506)
 Coordinates:
     latitude   (space) float64 56.12
     longitude  (space) float64 11.38
   * time       (time) datetime64[ns] 2007-01-02 ... 2020-01-01T01:00:00
     space_id   (space) int64 0
 Dimensions without coordinates: space
 Data variables:
     STATE1     (space, time) float64 0.07079 0.05532 0.04846 ... 0.06611 0.06511
     STATE2     (space, time) float64 0.04366 0.0462 0.03821 ... 0.0361 0.05612
     STATE3     (space, time) float64 280.0 270.4 285.5 277.4 ... 282.4 278.7
     STATE4     (space, time) float64 274.8 278.4 280.6 283.7 ... 281.2 281.9
     STATE5     (space, time) float64 280.9 279.7 278.0 278.0 ... 276.1 279.1
     OBS        (space, time) float64 -9.49 -8.494 -9.069 ... -8.1721 -8.157
 Attributes:
     license:  data license
     source:   data source,)
```

### 1-D split by indetifier
```python
identifier = {"space": np.array([0, 0, 1, 1, 1])}
bags = motrainer.dataset_split(ds, identifier)

ds_splitted = bags.compute()
print(ds_splitted[0].dims['space'])
print(ds_splitted[1].dims['space'])
```

```output
2
3
```

### 2-D split by indetifier
```python
id_time = np.zeros(8506)
id_time[2000:]=1
identifier = {"space": np.array([0, 1, 1, 0, 1]), "time": id_time}
bags = motrainer.dataset_split(ds, identifier)

print(bags)
```
```output
dask.bag<from_sequence, npartitions=4>
```

```python
for parts in ds_splitted:
    print(parts.dims)
```
```output
Frozen({'samples': 4000})
Frozen({'samples': 13012})
Frozen({'samples': 6000})
Frozen({'samples': 19518})
```

## Train-Test Split

### Split by coordinates
```python
train_test_bags = bags.map(
    motrainer.train_test_split, split={"time": np.datetime64("2017-01-01")}
)
```

```python
train_bags = train_test_bags.pluck(0)
test_bags = train_test_bags.pluck(1)
```

```python
motrainer.train_test_split(ds, split={"time": np.datetime64("2017-01-01")})
```

```python
motrainer.train_test_split(ds, split={"time": np.datetime64("2017-01-01")}, reverse=True)
```

### Split by mask

```python
mask = ds_valid["time"] < np.datetime64("2017-01-01")
train, test = train_test_split(ds_valid, mask=mask)
```

```python
mask = ds_valid["time"] < np.datetime64("2017-01-01")
train, test = train_test_split(ds_valid, mask=mask, reverse=True)
```

## Prepare data for training


## Parallel ML Model Optimization with Dask-ml

### Extract training data
```python
def to_dataframe(ds):
    return ds.to_dask_dataframe()

def chunk(ds, chunks):
    return ds.chunk(chunks)
```

```python
train_bags = train_test_bags.pluck(0).map(chunk, {"space": 500}).map(to_dataframe)
```

### Setup searching grid
```python
import dask_ml.model_selection as dcv
from sklearn import svm, datasets

parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
svc = svm.SVC()
search = dcv.GridSearchCV(svc, parameters)
```

### Grid search in parallel
```python
def optimize(df, grid_search, input_list, output_list):
    """Customized Optimization Function
    """
    df = df.dropna()
    grid_result = grid_search.fit(df[input_list], df[output_list])
    return grid_result
```

```python
input_list = ["STATE1", "STATE2", "STATE3", "STATE4", "STATE5"]
output_list = ["slop"]
optimazed_estimators = train_bags.map(
    optimize, search, input_list, output_list
)
```

## Parallel DNN Training with JackknifeGPI

### Train one grid point
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


