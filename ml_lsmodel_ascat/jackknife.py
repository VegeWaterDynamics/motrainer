import numpy as np
import pickle
import sklearn
import logging
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from ml_lsmodel_ascat.dnn import NNTrain
from ml_lsmodel_ascat.util import performance, normalize

logger = logging.getLogger(__name__)


class JackknifeGPI(object):
    def __init__(self,
                 gpi_data,
                 val_split_year,
                 input_list,
                 output_list,
                 export_all_years=True,
                 outpath='./jackknife_results'):

        logger.info('Initializing Jackkinfe trainning:\n'
                    'val_split_year: {}\n'
                    'input_list: {}\n'
                    'output_list: {}\n'.format(val_split_year, input_list,
                                               output_list))

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
              performance_method='rmse',
              training_method='dnn',
              verbose=0):

        # Data normalization
        logger.debug('Normalizing input/output data. Method: {}.'.format(
            normalize_method))
        self.gpi_data[self.input_list], scaler_input = normalize(
            self.gpi_data[self.input_list], normalize_method)
        self.gpi_data[self.output_list], scaler_output = normalize(
            self.gpi_data[self.output_list], normalize_method)

        # Data split
        logger.debug(
            'Spliting Trainning and validation data. Split year: {}.'.format(
                self.val_split_year))
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

            logger.info('Jackknife on year: {}.'.format(str(this_year)))

            train_all = jackknife_all[(jackknife_all.index.year != this_year)]
            test_all = jackknife_all[(jackknife_all.index.year == this_year)]
            train_input, train_output = train_all[
                self.input_list].values, train_all[self.output_list].values
            test_input, test_output = test_all[
                self.input_list].values, test_all[self.output_list].values

            # Execute training
            training = NNTrain(train_input, train_output)

            # Set searching space
            training.update_space(**searching_space)

            # Optimization
            training.optimize(**optimize_space,
                              training_method=training_method,
                              verbose=verbose)

            # TODO: Add warning if no model selected for the year
            if training.model is None:
                logger.warning('No best model was found for year: {}.'.format(
                    str(this_year)))
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
        logger.info('Found best year: {}'
                    'A-priori performance: {}'
                    'Post-priori performance: {}'.format(
                        str(self.best_year), self.apr_perf, self.post_perf))

    def export_best(self, output_options=['model', 'hyperparameters']):

        logger.info(
            'Exporting model and hyperparameters of year {} to {}'.format(
                self.best_year, self.outpath))

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
