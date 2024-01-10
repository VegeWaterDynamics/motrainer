import unittest

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from motrainer import util


def init_trainning():
    x = np.linspace(-0.5, 0.5, 100)
    y = np.linspace(-0.5, 0.5, 100)
    datain = np.column_stack((x, x))
    dataout = np.column_stack((y, y))
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5, input_shape=(2, )))
    model.add(tf.keras.layers.Dense(units=5))
    model.add(tf.keras.layers.Dense(units=2))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae'])
    model.fit(x=datain, y=dataout, epochs=20, verbose=0, validation_split=0.2)
    return model


class TestUtil(unittest.TestCase):
    model = init_trainning()

    def test_performance(self):
        x = np.linspace(-0.5, 0.5, 1000)
        y = np.linspace(-0.5, 0.5, 1000)
        datain = np.column_stack((x, x))
        data_label = np.column_stack((y, y))

        for method in ['rmse', 'mae', 'pearson', 'spearman']:
            perf = util.performance(datain, data_label, TestUtil.model, method)
            self.assertIsNotNone(perf)

    def test_normalize_standard(self):
        x = np.linspace(-0.5, 0.5, 200) + 1
        datain = np.column_stack((x, x))
        data_norm, scaler = util.normalize(datain, 'standard')
        self.assertAlmostEqual(data_norm.mean(), 0.0, 10)
        self.assertIsNotNone(scaler)

    def test_normalize_minmax(self):
        x = np.linspace(-0.5, 0.5, 200) + 1
        datain = np.column_stack((x, x))
        data_norm, scaler = util.normalize(datain, 'min_max')
        self.assertAlmostEqual(data_norm.min(), 0.0, 10)
        self.assertIsNotNone(scaler)


@pytest.fixture
def sklearn_model():
    # create a simple sklearn model
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5],
        'target': [10, 20, 30, 40, 50]
        })
    model = LinearRegression()
    model.fit(df[['feature']], df['target'])
    return model


class TestUtilSklearnIO:

    def test_sklearn_io(self, sklearn_model, tmp_path):
        model_path = tmp_path / 'test.h5'
        util.sklearn_save(sklearn_model, model_path, meta_data={'test': 1})

        # load the model
        model, metadata = util.sklearn_load(model_path)

        # check the model
        assert isinstance(model, LinearRegression)
        # check the metadata
        assert metadata['test'] == 1
