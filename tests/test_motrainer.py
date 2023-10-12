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


class TestModelSplit:
    @pytest.fixture
    def ds(self):
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

    def test_split_space(self, ds):
        identifier = {"space": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])}  # 4 and 6
        db = ds.mot.dataset_split(identifier)
        list_ds = db.compute()
        space_lens = [len(list_ds[i].space) for i in range(2)]
        assert len(list_ds) == 2
        assert set(space_lens).issubset(
            [4, 6]
        )  # Check the size of separated is 4 and 6, regardless of order

    def test_split_time(self, ds):
        identifier = {"time": np.array([1, 0, 0, 0, 0])}  # 1 and 4
        db = ds.mot.dataset_split(identifier)
        list_ds = db.compute()
        time_lens = [len(list_ds[i].time) for i in range(2)]
        assert len(list_ds) == 2
        assert set(time_lens).issubset(
            [1, 4]
        )  # Check the size of separated is 1 and 4, regardless of order

    def test_split_2d(self, ds):
        identifier = {
            "space": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),  # 4 and 6
            "time": np.array([1, 0, 0, 0, 0]),  # 1 and 4
        }
        db = ds.mot.dataset_split(identifier)
        list_ds = db.compute()
        samples_lens = [len(list_ds[i].samples) for i in range(4)]
        assert len(list_ds) == 4  # In total 4 ds
        assert set(samples_lens).issubset(
            [4, 16, 6, 24]
        )  # Check sample lenth regardless of order
