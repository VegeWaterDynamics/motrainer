import warnings
import xarray as xr
import numpy as np
import dask.bag as db

MOT_DIMS = ["space", "time"]  # Expected xr.Dataset dimensions


@xr.register_dataset_accessor("mot")
class MOTrainerDataset:
    """Xarray.Dataset extension for MOTrainer

    It contains methods facilitating distributed training tasks. 

    Following constraints are applied to a regular Dataset:
        1. The Dataset should have exactly two dimensions: "space" and "time"
        2. There are no duplicated entries in coordinates
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def is_valid(self) -> bool:
        """Check if a Dataset is valid for MOTrainer.
        The following checks will be applied:
            - The Dastaset has exactly 2 dimensions
            - The 2 dims are "space" and "time"
            - There are no duplicated coordinates
        A UserWarning will be raised for each failed check.

        Returns:
            bool: Result of check in Boolean.
                  If all checks are passed, it will be True. Otherwise False.
        """

        flag_valid = True

        # Dimension size should be 2
        if len(self._obj.dims) != 2:
            warnings.warn(
                f'Dataset should have two dimensions: "space" and "time"', UserWarning
            )
            flag_valid = False

        # space and time dimensions should exist
        for dim in MOT_DIMS:
            if not (dim in self._obj.dims):
                warnings.warn(f"{dim} not found in the dimensions", UserWarning)
                flag_valid = False

        # Check duplicated coordinates
        for coord in self._obj.coords:
            if np.unique(self._obj[coord]).shape != self._obj[coord].shape:
                warnings.warn(f"Duplicated coordinates found in {coord}", UserWarning)
                flag_valid = False

        return flag_valid

    def model_split(self, identifier: dict | str):
        """Split a Dataset by indentifier for independent training tasks.

        Args:
            identifier (dict | str): Split identifier. 
            When `indentifier` is a dictionary, it should map "space" and/or "time" dimension with corresponding separation indentifier.
            When `indentifier` is a string, `model_split` will use the corresponding field as the indertifier. This field can only be 1D.

        Returns:
            dask.bag: A Dask Databag of splited Datasets 
        """

        if isinstance(identifier, dict):
            if len(identifier.keys())==0:
                raise ValueError('identifier is empty')
            if not set(identifier.keys()).issubset(MOT_DIMS):
                raise ValueError('Acceptable keys are "space" and/or "time".')
        elif isinstance(identifier, str):
            if not (identifier in self._obj.variables):
                raise ValueError(f'Cannot find "{identifier}" in the Dataset')
            if len(self._obj[identifier].dims)>1:
                raise ValueError(f'Field "{identifier}" is not 1D. To perform 2D split, please specify both space and time indertifier in a dict.')
            identifier = {identifier: self._obj[identifier].values}
        else:
            raise NotImplementedError("identifier must be a dictionary or string")
        
        ds = self._obj

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
            ds = (
                ds.reset_index(MOT_DIMS)
                .stack(
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
