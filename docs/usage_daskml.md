# Parallel ML Model Optimization with Dask-ml

## Extract training data
```python
def to_dataframe(ds):
    return ds.to_dask_dataframe()

def chunk(ds, chunks):
    return ds.chunk(chunks)
```

```python
train_bags = train_test_bags.pluck(0).map(chunk, {"space": 500}).map(to_dataframe)
```

## Setup searching grid
```python
import dask_ml.model_selection as dcv
from sklearn import svm, datasets

parameters = {'kernel': ['linear', 'rbf'], 'C': [1, 10]}
svc = svm.SVC()
search = dcv.GridSearchCV(svc, parameters)
```

## Grid search in parallel
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