# Data Split

This section demonstrates how to use motrainer to partition data for parallel ML model training.

We start with an [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object, which contains spatio-temporal data for training multiple independent Machine Learning (ML) models. Examples of reading spatio-temporal data into `xarray.Dataset` can be found in [Example 1](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_one_df/) and [Example 2](https://vegewaterdynamics.github.io/motrainer/notebooks/example_read_from_multiple_df/).

The example Dataset object `ds` contains input and output data from 5 grid cells. It has two dimensions: `space` and `time`. It has six data variables, where `STATE1` to `STATE5` are physical model states (input data) and observations (output data).

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

Before splitting, you can verify if a dataset is splittable using the `is_splitable` function:

```python
motrainer.is_splitable(ds)
```

```output
True
```

The is_splitable function will return True if the dataset has exactly two dimensions: "space" and "time", and there are no duplicated keys in any of the coordinates of the dataset.

## Splitting Spatio-Temporal Dataset for Independent Training Processes

The dataset_split function can be used to partition data. The split can be performed by specifying a dimension name or using an identifier.

### Splitting by Dimension Names

One can specify adimension name to split, e.g. "space":

```python
import motrainer
bags = motrainer.dataset_split(ds, "space")
print(bags)
```
```output
dask.bag<from_sequence, npartitions=5>
```

This will split `ds` per grid cell in to a [`Dask.bag`](https://docs.dask.org/en/stable/bag.html) object. Each partition is an independent gridcell. 

We can check one grid cell by:

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

One can also create an identifier dictionary to split. The keys of the dictionary should be a subset of {"space", "time"}, mapping "space" and/or "time" dimension with corresponding separation identifier.

For example, `ds` can be splitted into two parts (first+fourth grid, and the rest) in space:

```python
import numpy as np

identifier = {"space": np.array([0, 1, 1, 0, 1])}
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

One can perform a 2-D split by providing identifiers in both space and time dimensions:

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

Before training, one can further split the datasets into training and testing datasets to reserve some data for testing. If the split needs to be based on the space and time coordinates, one can use `motrainer.train_test_split` to perform this split. Otherwise we recommend `sklearn.model_selection.train_test_split`(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) for simplicity.

### Splitting by Coordinates

When specifying a space or time coordinate, the dataset will be split into two parts: the part smaller ("<") than the coordinates and the rest (">="). By default, the former will be the training data and the latter will be testing.

If the dataset has already be splitted into `Dask.bags`, we recommend to use the [`map`](https://docs.dask.org/en/stable/generated/dask.bag.map.html#dask.bag.map) function to apply train-test split to each splitted element.

The following will select data before 2017-01-01 as training data:

```python
train_test_bags = bags.map(
    motrainer.train_test_split, split={"time": np.datetime64("2017-01-01")}
)
```

Then extract train and test data using `pluck`:

```python
train_bags = train_test_bags.pluck(0)
test_bags = train_test_bags.pluck(1)
```

When `reverse=True` is present, the latter part (">=" coordinate) will be training data, the rest will be testing. The following code will select data after (and include) 2017-01-01 as training data:

```python
train_test_bags = bags.map(
    motrainer.train_test_split, split={"time": np.datetime64("2017-01-01"), reverse=True}
)
train_bags = train_test_bags.pluck(0)
test_bags = train_test_bags.pluck(1)
```


One can also apply `motrainer.train_test_split` directly to an `xarray.Dataset` object:
```python
motrainer.train_test_split(ds, split={"time": np.datetime64("2017-01-01")})
```

### Splitting by Mask

Alternatively, you can also initiate a `mask` to perform training data. By default, training data will be where `mask` is `True`. For example, if you would like to have data before 2017-01-01 as training data:

```python
mask = ds_valid["time"] < np.datetime64("2017-01-01")
train, test = train_test_split(ds_valid, mask=mask)
```

If `reverse` is specified, training data will be where mask is `False`. The following will select data after (and include) 2017-01-01 as training data:

```python
mask = ds_valid["time"] < np.datetime64("2017-01-01")
train, test = train_test_split(ds_valid, mask=mask, reverse=True)
```
