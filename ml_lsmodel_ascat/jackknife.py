import numpy as np
import pickle
import sklearn
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from ml_lsmodel_ascat.dnn import NNTrain
from ml_lsmodel_ascat.util import performance, normalize


class JackknifeGPI(object):
    def __init__(self,
                 gpi_data,
                 val_split_year,
                 input_list,
                 output_list,
                 export_all_years=True,
                 outpath='./jackknife_results'):
        self.gpi_data = gpi_data
        self.input_list = input_list
        self.output_list = output_list
        self.gpi_input = gpi_data[input_list]
        self.gpi_output = gpi_data[output_list]
        self.val_split_year = val_split_year
        self.export_all_years = export_all_years
        self.outpath = outpath
        Path(self.outpath).parent.mkdir(parents=True, exist_ok=True)

        assert not (
            self.gpi_data.isnull().values.any()), 'Nan value(s) in gpi_data!'

    def train(self,
              searching_space,
              optimize_space,
              normalize_method='standard',
              training_method='dnn',
              performance_method='rmse',
              val_split_year=2017):

        # Data normalization
        self.gpi_data[self.input_list], scaler_input = normalize(
            self.gpi_data[self.input_list], normalize_method)
        self.gpi_data[self.output_list], scaler_output = normalize(
            self.gpi_data[self.output_list], normalize_method)

        # Data split
        jackknife_all = self.gpi_data[
            self.gpi_data.index.year < self.val_split_year]
        year_list = jackknife_all.copy().resample('Y').mean().index.year
        vali_all = self.gpi_data[
            self.gpi_data.index.year >= self.val_split_year]
        vali_input = vali_all[self.input_list].values
        vali_output = vali_all[self.output_list].values

        # Jackknife in time
        loo = LeaveOneOut()
        best_perf_sum = None
        for train_index, test_index in loo.split(year_list):
            this_year = test_index[0] + year_list[0]

            print('=====================================')
            print('jackknife on ' + str(this_year))
            print('=====================================')

            train_all = jackknife_all[(jackknife_all.index.year != this_year)]
            test_all = jackknife_all[(jackknife_all.index.year == this_year)]
            train_input, train_output = train_all[
                self.input_list].values, train_all[self.output_list].values
            test_input, test_output = test_all[
                self.input_list].values, test_all[self.output_list].values

            # Execute training
            training = NNTrain(train_input, train_output)

            # Set searching space
            training.update_space(learning_rate=[
                searching_space['learning_rate'][0],
                searching_space['learning_rate'][1]
            ],
            num_dense_layers=[
                    searching_space['num_dense_layers'][0],
                    searching_space['num_dense_layers'][1]
            ],
            num_input_nodes=[
                    searching_space['num_input_nodes'][0],
                    searching_space['num_input_nodes'][1]
            ],
            num_dense_nodes=[
                    searching_space['num_dense_nodes'][0],
                    searching_space['num_dense_nodes'][1]
            ],
            batch_size=[
                    searching_space['batch_size'][0],
                    searching_space['batch_size'][1]
            ],
            activation=searching_space['activation'])

            # Optimization
            training.optimize(
                best_loss=optimize_space['best_loss'],
                n_calls=optimize_space['n_calls'],
                noise=optimize_space['noise'],
                n_jobs=optimize_space['n_jobs'],
                kappa=optimize_space['kappa'],
                validation_split=optimize_space['validation_split'],
                x0=optimize_space['x0'],
                training_method='dnn')

            # TODO: Add warning if no model selected for the year
            if training.model is None:
                continue

            if self.export_all_years:
                path_model = '{}/all_years/optimized_model_{}'.format(
                    self.outpath, this_year)
                path_hyperparas = '{}/all_years/hyperparameters_{}'.format(
                    self.outpath, this_year)
                training.export(path_model=path_model,
                                path_hyperparameters=path_hyperparas)

            # find minimum rmse
            # TODO: mae, pearson, spearman
            apr_perf = performance(test_input, test_output, training.model,
                                   performance_method, scaler_output)
            perf_sum = np.nansum(apr_perf)
            if best_perf_sum is None:
                best_perf_sum = perf_sum
            if perf_sum <= best_perf_sum:
                self.apr_perf = apr_perf
                self.post_perf = performance(vali_input, vali_output,
                                             training.model,
                                             performance_method, scaler_output)
                self.best_train = training
                self.best_year = this_year

    def export_best(self, output_options=['model', 'hyperparameters']):

        if 'model' in output_options:
            path_model = '{}/best_optimized_model_{}'.format(
                self.outpath, self.best_year)
        else:
            path_model = None

        if 'hyperparameters' in output_options:
            path_hyperparameters = '{}/best_hyperparameters_{}'.format(
                self.outpath, self.best_year)
        else:
            path_hyperparameters = None

        self.best_train.export(path_model=path_model,
                               path_hyperparameters=path_hyperparameters)
