
import h5py
import numpy as np
import pandas as pd
import pytest

from motrainer.jackknife import JackknifeGPI


@pytest.fixture
def train_data():
    start_year = 2017
    end_year = 2020
    obs_per_year = 20
    x = np.tile(np.linspace(-0.5, 0.5, obs_per_year), end_year - start_year)
    y = x
    df = pd.DataFrame(
        data={"x": x, "y": y},
        index=pd.date_range(
            start=f"1/1/{start_year}",
            end=f"1/1/{end_year}",
            periods=obs_per_year * (end_year - start_year),
        ),
    )
    return df


class TestJacknife:
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_initialize_gpi(self, train_data):
        gpi = JackknifeGPI(
            train_data, val_split_year=2019, input_list=["x"], output_list=["y"]
        )
        assert gpi.gpi_input["x"] is not None
        assert gpi.gpi_output["y"] is not None

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.fixture
    def test_gpi_results_exist(self, train_data, tmp_path):
        gpi = JackknifeGPI(
            train_data,
            val_split_year=2019,
            input_list=["x"],
            output_list=["y"],
            export_all_years=False,
            outpath=tmp_path,
        )
        gpi.train(
            searching_space={
                "learning_rate": [0.01, 0.02],
                "num_dense_layers": [1, 2],
                "num_input_nodes": [5, 6],
                "num_dense_nodes": [5, 6],
            },
            optimize_space={
                "best_loss": 10,
                "epochs": 1,
                "x0": [0.01, 1, 5, 5, "relu", 64],
            },
            normalize_method="standard",
            training_method="dnn",
            performance_method="rmse",
            verbose=0,
        )
        gpi.export_best()
        assert gpi.apr_perf is not None
        assert gpi.post_perf is not None
        assert gpi.best_train is not None
        assert gpi.best_year in [2017, 2018]

        # test meta_data saved in hdf5 file
        with h5py.File(tmp_path / "best_optimized_model_2017.h5", 'r') as f:
            best_year = f.attrs['best_year']
            input_list = f.attrs['input_list']

        assert best_year == 2017
        assert input_list == gpi.input_list
