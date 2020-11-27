import unittest
import datetime
import pandas as pd
import numpy as np
from ml_lsmodel_ascat.jackknife import JackknifeGPI


def init_gpi_linear():
    start_year = 2017
    end_year = 2020
    obs_per_year = 20
    x = np.tile(np.linspace(-0.5, 0.5, obs_per_year), end_year - start_year)
    y = x
    df = pd.DataFrame(data={
        'x': x,
        'y': y
    },
                      index=pd.date_range(start='1/1/{}'.format(start_year),
                                          end='1/1/{}'.format(end_year),
                                          periods=obs_per_year *
                                          (end_year - start_year)))
    return df


class TestJacknife(unittest.TestCase):
    def test_initialize_gpi(self):
        gpi_data = init_gpi_linear()
        gpi = JackknifeGPI(gpi_data,
                           val_split_year=2019,
                           input_list=['x'],
                           output_list=['y'])
        self.assertAlmostEqual(gpi.gpi_input['x'].sum(), gpi_data['x'].sum(),
                               1e-10)
        self.assertAlmostEqual(gpi.gpi_output['y'].sum(), gpi_data['y'].sum(),
                               1e-10)

    def test_gpi_results_exist(self):
        gpi_data = init_gpi_linear()
        gpi = JackknifeGPI(gpi_data,
                           val_split_year=2019,
                           input_list=['x'],
                           output_list=['y'])
        gpi.train(searching_space={
            'learning_rate': [0.01, 0.02],
            'num_dense_layers': [1, 2],
            'num_input_nodes': [5, 6],
            'num_dense_nodes': [5, 6],
        },
                  optimize_space={
                      'best_loss': 10,
                      'epochs': 1,
                      'x0': [0.01, 1, 5, 5, 'relu', 64]
                  },
                  normalize_method='standard',
                  training_method='dnn',
                  performance_method='rmse',
                  verbose=0)
        self.gpi = gpi
        self.assertIsNotNone(self.gpi.apr_perf)
        self.assertIsNotNone(self.gpi.post_perf)
        self.assertIsNotNone(self.gpi.best_train)
        self.assertIn(self.gpi.best_year, [2017, 2018])
