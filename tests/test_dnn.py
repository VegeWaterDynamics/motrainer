import unittest
import datetime
import pandas as pd
import numpy as np
from ml_lsmodel_ascat.jackknife import JackknifeGPI
from ml_lsmodel_ascat.dnn import NNTrain


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
                               10)
        self.assertAlmostEqual(gpi.gpi_output['y'].sum(), gpi_data['y'].sum(),
                               10)

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


class TestDNN(unittest.TestCase):
    def test_init_train(self):
        datain = pd.DataFrame(data={
            'x1': np.linspace(0, 0.5, 100),
            'x2': np.linspace(0, 0.5, 100)
        })
        dataout = pd.DataFrame(data={'y': np.linspace(0, 1, 100)})
        test_train = NNTrain(datain, dataout)
        self.assertEqual((test_train.train_input - datain).values.sum(), 0.)

    def test_update_space(self):
        datain = pd.DataFrame(data={
            'x1': np.linspace(0, 0.5, 100),
            'x2': np.linspace(0, 0.5, 100)
        })
        dataout = pd.DataFrame(data={'y': np.linspace(0, 1, 100)})
        test_train = NNTrain(datain, dataout)
        searching_space = {
            'learning_rate': [0.01, 0.02],
            'num_dense_layers': [1, 2],
            'num_input_nodes': [5, 6],
            'num_dense_nodes': [5, 6],
        }
        test_train.update_space(**searching_space)
        for key, value in searching_space.items():
            self.assertListEqual([
                test_train.dimensions[key].low, test_train.dimensions[key].high
            ], value)

    def test_dnn_results_exist(self):
        datain = pd.DataFrame(data={
            'x1': np.linspace(0, 0.5, 100),
            'x2': np.linspace(0, 0.5, 100)
        })
        dataout = pd.DataFrame(data={'y': np.linspace(0, 1, 100)})
        test_train = NNTrain(datain, dataout)
        test_train.update_space(learning_rate=[0.01, 0.02],
                                num_dense_layers=[1, 2],
                                num_input_nodes=[5, 6],
                                num_dense_nodes=[5, 6])
        test_train.optimize(best_loss=10,
                            epochs=1,
                            x0=[0.01, 2, 5, 5, 'relu', 64])
        self.assertIsNotNone(test_train.model)
        self.assertIsNotNone(test_train.best_loss)

    def test_dnn_lossweight_loss_weighted(self):
        datain = pd.DataFrame(data={
            'x1': np.linspace(0, 0.5, 100),
            'x2': np.linspace(0, 0.5, 100)
        })
        dataout = datain
        test_train = NNTrain(datain, dataout)
        test_train.update_space(learning_rate=[0.01, 0.02],
                                num_dense_layers=[1, 2],
                                num_input_nodes=[5, 6],
                                num_dense_nodes=[5, 6])
        test_train.optimize(best_loss=10,
                            epochs=1,
                            training_method='dnn_lossweights',
                            loss_weights=[1, 0.5],
                            x0=[0.01, 2, 5, 5, 'relu', 64])
        test_train.model.predict(datain)
        ls1 = test_train.history['loss'][0]
        ls2 = test_train.history['out1_loss'][
            0] + test_train.history['out2_loss'][0] * 0.5
        self.assertAlmostEqual(ls1, ls2, 5)
