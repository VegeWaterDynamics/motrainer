import h5py
import numpy as np
import pandas as pd
import pytest
from skopt.space import Real

from motrainer.dnn import NNTrain


@pytest.fixture
def train_data():
    start_year = 2017
    end_year = 2018
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


# Expected searching dimensions
EXPECTED_DIMS = {
    "learning_rate",
    "num_dense_layers",
    "num_input_nodes",
    "num_dense_nodes",
    "activation",
    "batch_size",
}


class TestDNN:
    def test_init_train(self, train_data):
        test_train = NNTrain(train_data["x"], train_data["y"])
        assert test_train.train_input.equals(train_data["x"])
        assert test_train.train_output.equals(train_data["y"])
        assert EXPECTED_DIMS.issubset(test_train.dimensions.keys())


class TestUpdateSpace:
    def test_update_space_list(self, train_data):
        test_train = NNTrain(train_data["x"], train_data["y"])
        searching_space = {
            "learning_rate": [0.01, 0.02],
            "num_dense_layers": [1, 2],
        }
        test_train.update_space(**searching_space)

        for key, value in searching_space.items():
            assert key in test_train.dimensions.keys()
            assert value == searching_space[key]

    def test_update_space_skopt(self, train_data):
        nlayer = Real(low=1, high=2, prior="log-uniform", name="num_dense_layers")
        searching_space = {
            "num_dense_layers": nlayer,
        }

        test_train = NNTrain(train_data["x"], train_data["y"])
        test_train.update_space(**searching_space)

        assert "num_dense_layers" in test_train.dimensions.keys()
        assert test_train.dimensions["num_dense_layers"].low == 1
        assert test_train.dimensions["num_dense_layers"].high == 2


class TestOptimize:
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.fixture
    def test_dnn_results_exist(self, tmp_path):
        datain = pd.DataFrame(
            data={"x1": np.linspace(0, 1, 10), "x2": np.linspace(0, 1, 10)}
        )
        dataout = pd.DataFrame(data={"y": np.linspace(0, 1, 100)})
        test_train = NNTrain(datain, dataout)
        test_train.update_space(
            learning_rate=[0.01, 0.02],
            num_dense_layers=[1, 2],
            num_input_nodes=[5, 6],
            num_dense_nodes=[5, 6],
        )
        test_train.optimize(best_loss=10, epochs=1, x0=[0.01, 2, 5, 5, "relu", 64])
        expected_hyperparameters = sorted(
            zip(
                test_train.gp_result.func_vals,
                test_train.gp_result.x_iters,
                strict=True
                )
        )
        test_train.export(f"{tmp_path}/model")
        assert test_train.model is not None

        # load hyperparameters from "model.h5" and assert they are the same
        with h5py.File(f"{tmp_path}/model.h5", 'r') as f:
            param = f.attrs['hyperparameters']
        # convert hyperparameters to a list of tuples
        hyperparameters = eval(param)
        assert hyperparameters[0][0] == expected_hyperparameters[0][0]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
    def test_dnn_results_exist_lossweight(self):
        datain = pd.DataFrame(
            data={"x1": np.linspace(0, 1, 10), "x2": np.linspace(0, 1, 10)}
        )
        dataout = datain
        test_train = NNTrain(datain, dataout)
        test_train.update_space(
            learning_rate=[0.01, 0.02],
            num_dense_layers=[1, 2],
            num_input_nodes=[5, 6],
            num_dense_nodes=[5, 6],
        )
        test_train.optimize(
            best_loss=10,
            epochs=1,
            training_method="dnn_lossweights",
            loss_weights=[1, 0.5],
            x0=[0.01, 2, 5, 5, "relu", 64],
        )

        assert test_train.model is not None
