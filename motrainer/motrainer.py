import warnings
import xarray as xr
import numpy as np

DIMS = ["space", "time"]  # Expected xr.Dataset dimensions


@xr.register_dataset_accessor("mot")
class MOTrainerDataset:
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
                f'Dataset should have two dims: "space" and "time"', UserWarning
            )
            flag_valid = False

        # space and time dimensions should exist
        for dim in DIMS:
            if not (dim in self._obj.dims):
                warnings.warn(f"{dim} not found in the dimensions", UserWarning)
                flag_valid = False

        # Check duplicated coordinates
        for coord in self._obj.coords:
            if np.unique(self._obj[coord]).shape != self._obj[coord].shape:
                warnings.warn(f"Duplicated coordinates found in {coord}", UserWarning)
                flag_valid = False

        return flag_valid

    def model_split(self, identifier: dict = {}, chunks: tuple = ()):
        # if not(identifier) and not(chunks):
        #     raise ValueError('Need to specify "identifier" or chunks')

        pass
