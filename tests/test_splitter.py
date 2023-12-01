import xarray as xr
import numpy as np
import pytest
from motrainer import is_splitable, dataset_split
from motrainer.splitter import _regulate_identifier


@pytest.fixture
def ds_valid():
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


class TestValidate:
    def test_valid_ds(self, ds_valid):
        assert is_splitable(ds_valid)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_miss_one_dim(self, ds_valid):
        ds_invalid = ds_valid.drop_dims("space")
        assert not is_splitable(ds_invalid)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_redundant_dim(self, ds_valid):
        ds_invalid = ds_valid.expand_dims({"redundant": 3})
        assert not is_splitable(ds_invalid)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_duplicate_coords(self, ds_valid):
        ds_invalid = ds_valid
        ds_invalid["space"] = np.ones(ds_valid.dims["space"])
        assert not is_splitable(ds_invalid)


class TestModelSplit:
    def test_split_space(self, ds_valid):
        identifier = {"space": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])}  # 4 and 6
        db = dataset_split(ds_valid, identifier)
        list_ds = db.compute()
        space_lens = [len(list_ds[i].space) for i in range(2)]
        assert len(list_ds) == 2
        assert set(space_lens).issubset(
            [4, 6]
        )  # Check the size of separated is 4 and 6, regardless of order

    def test_split_time(self, ds_valid):
        identifier = {"time": np.array([1, 0, 0, 0, 0])}  # 1 and 4
        db = dataset_split(ds_valid, identifier)
        list_ds = db.compute()
        time_lens = [len(list_ds[i].time) for i in range(2)]
        assert len(list_ds) == 2
        assert set(time_lens).issubset(
            [1, 4]
        )  # Check the size of separated is 1 and 4, regardless of order

    def test_split_2d(self, ds_valid):
        identifier = {
            "space": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),  # 4 and 6
            "time": np.array([1, 0, 0, 0, 0]),  # 1 and 4
        }
        db = dataset_split(ds_valid, identifier)
        list_ds = db.compute()
        samples_lens = [len(list_ds[i].samples) for i in range(4)]
        assert len(list_ds) == 4  # In total 4 ds
        assert set(samples_lens).issubset(
            [4, 16, 6, 24]
        )  # Check sample lenth regardless of order

    def test_split_by_dim(self, ds_valid):
        db_space = dataset_split(ds_valid, "space")
        db_time = dataset_split(ds_valid, "time")
        list_space = db_space.compute()
        list_time = db_time.compute()
        assert len(list_space) == 10
        assert len(list_time) == 5

    def test_split_space_no_coords(self, ds_valid):
        db_no_space_coords = dataset_split(ds_valid.drop_vars("space"), "space")
        db_no_time_coords = dataset_split(ds_valid.drop_vars("time"), "time")
        list_ds_no_space = db_no_space_coords.compute()
        list_ds_no_time = db_no_time_coords.compute()
        assert len(list_ds_no_space) == 10
        assert len(list_ds_no_time) == 5


class TestValidateIdentifier:
    def test_dict_identifier(self, ds_valid):
        idf = _regulate_identifier(ds_valid, {"space": range(10)})
        assert idf == {"space": range(10)}

    def test_str_identifier(self, ds_valid):
        idf = _regulate_identifier(ds_valid, "space")
        assert idf == {"space": ds_valid["space"].values}

    def test_str_identifier_only_dims(self, ds_valid):
        ds = ds_valid.drop_vars("space")
        idf = _regulate_identifier(ds, "space")
        assert idf == {"space": range(ds.dims["space"])}

    def test_errors(self, ds_valid):
        with pytest.raises(ValueError):
            _ = _regulate_identifier(ds_valid, {})
        with pytest.raises(ValueError):
            _ = _regulate_identifier(
                ds_valid, {"space": range(10), "non_exists": range(10)}
            )
        with pytest.raises(ValueError):
            _ = _regulate_identifier(ds_valid, "non_exists")
        with pytest.raises(NotImplementedError):
            _ = _regulate_identifier(ds_valid, ["space"])
