import xarray as xr
import numpy as np
import pytest
import motrainer


class TestValidate:
    @pytest.fixture
    def ds_valid(self):
        nspace = 10
        ntime = 5
        return xr.Dataset(
            data_vars=dict(
                data=(
                    ["space", "time"],
                    np.arange(nspace * ntime).reshape((nspace, ntime)),
                ),
            ),
            coords=dict(
                space=(["space"], np.arange(nspace)),
                time=(["time"], np.arange(ntime)),
            ),
        )

    def test_valid_ds(self, ds_valid):
        assert ds_valid.mot.is_valid()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_miss_one_dim(self, ds_valid):
        ds_invalid = ds_valid.drop_dims("space")
        assert not ds_invalid.mot.is_valid()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_redundant_dim(self, ds_valid):
        ds_invalid = ds_valid.expand_dims({"redundant": 3})
        assert not ds_invalid.mot.is_valid()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_coords(self, ds_valid):
        ds_invalid = ds_valid
        ds_invalid["space"] = np.ones(ds_valid.dims["space"])
        assert not ds_invalid.mot.is_valid()
