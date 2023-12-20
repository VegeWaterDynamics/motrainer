import warnings

import dask.bag as db
import numpy as np
import xarray as xr

MOT_DIMS = ["space", "time"]  # Expected xr.Dataset dimensions


def is_splitable(ds: xr.Dataset) -> bool:
    """Check if a Dataset is can be splitted using MOTrainer.

    The following checks will be applied:
        - The Dastaset has exactly 2 dimensions
        - The 2 dims are "space" and "time"
        - There are no duplicated coordinates
    A UserWarning will be raised for each failed check.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray Dataset to be splitted.

    Returns
    -------
    bool
       Result of check in Boolean.
       If all checks are passed, it will be True. Otherwise False.
    """
    flag_valid = True

    # Dimension size should be 2
    if len(ds.dims) != 2:
        warnings.warn(
            'Dataset should have two dimensions: "space" and "time"',
            UserWarning,
            stacklevel=2,
        )
        flag_valid = False

    # space and time dimensions should exist
    for dim in MOT_DIMS:
        if dim not in ds.dims:
            warnings.warn(
                f"{dim} not found in the dimensions", UserWarning, stacklevel=2
            )
            flag_valid = False

    # Check duplicated coordinates
    for coord in ds.coords:
        if np.unique(ds[coord]).shape != ds[coord].shape:
            warnings.warn(
                f"Duplicated coordinates found in {coord}", UserWarning, stacklevel=2
            )
            flag_valid = False

    return flag_valid


def dataset_split(ds: xr.Dataset, identifier: dict | str) -> db:
    """Split a Dataset by indentifier for independent training tasks.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray Dataset to be splitted.
    identifier : dict | str
        When `indentifier` is a dictionary, its keys should be a subset of
        {"space", "time"},  and map "space" and/or "time" dimension with corresponding
        separation indentifier.

        When `indentifier` is a string, the separation will depends on if `indentifier`
        is a key of coords/data_vars or a dimension name ("space" or "time").
        In the former case the corresponding coords/data_vars will be used as separation
        indentifier.
        In the latter case ds will be separated per entry in that dimension.

    Returns
    -------
    dask.bag
        A Dask Databag of splited Datasets
    """
    identifier = _regulate_identifier(ds, identifier)

    list_id = []
    for key in MOT_DIMS:
        if key in identifier.keys():
            key_id = key + "_id"
            ds = ds.assign_coords({key_id: (key, identifier[key])})
            list_id.append(key_id)

    # Get the name of attributes to groupby
    if len(list_id) > 1:
        # Use space time cls coordinates as multi index
        # Must stack on MOT_DIMS to reduce dims of data variable
        multi_idx = ds.stack(samples=list_id).samples.values
        for dim in MOT_DIMS:
            if dim in ds.indexes:
                ds = ds.reset_index(dim)
        ds = (
            ds.stack(
                samples=MOT_DIMS, create_index=False
            )  # No index creation since this will be added next.
            .assign_coords(samples=multi_idx)
            .set_xindex(list_id)
        )
        key_gb = "samples"
    else:
        key_gb = list_id[0]

    # Groupby and separate to Dask Databags
    list_db = []
    for grp in list(ds.groupby(key_gb)):
        list_db.append(grp[1])
    bags = db.from_sequence(list_db)

    return bags


def train_test_split(
    ds: xr.Dataset,
    mask: xr.DataArray = None,
    split: dict = None,
    reverse: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Split data to train and test datasets.

    The split is performed either 1) by specifying the training data mask (`mask`)
    where training data locations are True, or 2) by a specifying a coordinate value
    (`split`) splitting the data into two.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset to split
    mask : xr.DataArray, optional
        Mask, True at training data locations. By default None
    split : dict, optional
        coordinate diactionary in {NAME: coordinates} which split the Dataset into two.
        The part smaller than it will be training, by default None.
    reverse : bool, optional
        Reverse the split results, by default False

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Split results. In (training, test).

    Raises
    ------
    ValueError
        When neither mask nor split is specified.
    ValueError
        When both mask and split are specified.
    """
    if mask is None and split is None:
        raise ValueError("Either mask or split should be specified.")
    elif mask is not None and split is not None:
        raise ValueError("Only one of mask and split should be specified.")

    # Convert split to mask
    if split is not None:
        _validate_train_test_split(split)
        mask = ds[list(split.keys())[0]] < list(split.values())[0]

    train = ds.where(mask, drop=True)
    test = ds.where(~mask, drop=True)

    return (test, train) if reverse else (train, test)


def _regulate_identifier(ds: xr.Dataset, identifier: dict | str) -> dict:
    """Regulate the split identifier w.r.t. the dataset.

    When `indentifier` is a dictionary:
        function will check if the keys are a subset of {"space", "time"}.

    When `indentifier` is a string:
        if `indentifier` is a key of any coords/data_vars, the corresponding
        coords/data_vars will be converter to a dict and used as indentifier
        if `indentifier` is not a key of any coords/data_vars, but is one of dimensions
        i.e. "space" or "time", then it will be converted to either:
            {"space": range(ds.dims["space"])} or
            {"time": range(ds.dims["space"])}

    Parameters
    ----------
    ds : xr.Dataset
        referce DS
    identifier : dict | str
        input indertifier

    Returns
    -------
    dict
        regulated identifier

    Raises
    ------
    ValueError
        identifier is an empty dict.
    ValueError
        identifier is a dict but has keys other that "space" and "time".
    ValueError
        identifier is a str but does not match any key in dims, coords or data_vars.
    NotImplementedError
        identifier is not a dict nor str
    """
    if isinstance(identifier, dict):
        if len(identifier.keys()) == 0:
            raise ValueError("identifier is empty")
        if not set(identifier.keys()).issubset(MOT_DIMS):
            raise ValueError('Acceptable keys are "space" and/or "time".')
    elif isinstance(identifier, str):
        if identifier in ds.variables:
            identifier = {identifier: ds[identifier].values}
        elif identifier in ds.dims:
            identifier = {identifier: range(ds.dims[identifier])}
        else:
            raise ValueError(f'Cannot find "{identifier}" in the Dataset')
    else:
        raise NotImplementedError("identifier must be a dictionary or string")

    return identifier


def _validate_train_test_split(split):
    if not isinstance(split, dict):
        raise ValueError("split should be a dict")

    if len(split.keys()) != 1:
        raise ValueError("split should only have 1 key")

    return None
