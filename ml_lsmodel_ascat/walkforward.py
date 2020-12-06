import numpy as np
import pickle
import sklearn
import logging
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit
from ml_lsmodel_ascat.dnn import NNTrain
from ml_lsmodel_ascat.util import performance, normalize, series_to_supervised

logger = logging.getLogger(__name__)


class ForwardChainingGPI(object):
    def __init__(self,
                 gpi_data,
                 val_split_year,
                 input_list,
                 output_list,
                 n_in, 
                 n_out,
                 export_all_years=True,
                 outpath='./jackknife_results'):

        logger.info('Initializing Jackkinfe trainning:\n'
                    'val_split_year: {}\n'
                    'input_list: {}\n'
                    'output_list: {}\n'.format(val_split_year, input_list,
                                               output_list))

        assert not (
            gpi_data.isnull().values.any()), 'Nan value(s) in gpi_data!'

        self.gpi_data = gpi_data
        self.input_list = input_list
        self.output_list = output_list
        self.gpi_input = gpi_data[input_list].copy()
        self.gpi_output = gpi_data[output_list].copy()
        self.val_split_year = val_split_year
        self.export_all_years = export_all_years
        self.outpath = outpath
        Path(self.outpath).parent.mkdir(parents=True, exist_ok=True)
        
        self.n_in = n_in
        self.n_out = n_out

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
        self.gpi_input[:], scaler_input = normalize(self.gpi_input,
                                                    normalize_method)
        self.gpi_output[:], scaler_output = normalize(self.gpi_output,
                                                      normalize_method)

        # Data split
        logger.debug(
            'Spliting Trainning and validation data. Split year: {}.'.format(
                self.val_split_year))
        jackknife_input = self.gpi_input[
            self.gpi_input.index.year < self.val_split_year]
        jackknife_output = self.gpi_output[
            self.gpi_output.index.year < self.val_split_year]
        vali_input = self.gpi_input[
            self.gpi_input.index.year >= self.val_split_year]
        vali_output = self.gpi_output[
            self.gpi_output.index.year >= self.val_split_year]
        year_list = jackknife_input.index.year.unique()
        
        vali_input, vali_output=series_to_supervised(vali_input,
                                                     vali_output,
                                                     self.input_list,
                                                     self.output_list,
                                                     n_in=self.n_in,
                                                     n_out=self.n_out,
                                                     dropnan=True)
        vali_input = vali_input.values.reshape(vali_input.shape[0],
                                               self.n_in,
                                               len(self.input_list))
        vali_output= vali_output.values.reshape(vali_output.shape[0],
                                                self.n_out,
                                                len(self.output_list)).squeeze()

        # Jackknife in time
#        loo = LeaveOneOut()
        tscv = TimeSeriesSplit()
        best_perf_sum = None
#        for train_index, test_index in loo.split(year_list):
        for train_index, test_index in tscv.split(year_list):
            test_year = test_index[0] + year_list[0]
            train_year = train_index+ year_list[0]
            
            logger.info(
                'Forward chaining training on year: {}.'.format(str(train_year)))
            logger.info(
                'Forward chaining testing on year: {}.'.format(str(test_year)))
#            train_input = jackknife_input[
#                jackknife_input.index.year != test_year]
            train_input = jackknife_input[
                jackknife_input.index.year.isin(list(train_year))]
#            train_output = jackknife_output[
#                jackknife_output.index.year != test_year]
            train_output = jackknife_output[
                jackknife_output.index.year.isin(list(train_year))]
            test_input = jackknife_input[jackknife_input.index.year ==
                                         test_year]
            test_output = jackknife_output[jackknife_output.index.year ==
                                           test_year]
            
            # Time series processing
            train_input, train_output=series_to_supervised(train_input,
                                                           train_output,
                                                           self.input_list, 
                                                           self.output_list,
                                                           n_in=self.n_in, 
                                                           n_out=self.n_out, 
                                                           dropnan=True)
            
            train_input = train_input.values.reshape(train_input.shape[0],
                                              self.n_in,
                                              len(self.input_list))
            train_output= train_output.values.reshape(train_output.shape[0],
                                              self.n_out,
                                              len(self.output_list)).squeeze()
            
            test_input, test_output=series_to_supervised(test_input,
                                                         test_output,
                                                         self.input_list,
                                                         self.output_list,
                                                         n_in=self.n_in,
                                                         n_out=self.n_out,
                                                         dropnan=True)
            test_input = test_input.values.reshape(test_input.shape[0],
                                                   self.n_in,
                                                   len(self.input_list))
            test_output= test_output.values.reshape(test_output.shape[0],
                                                    self.n_out,
                                                    len(self.output_list)).squeeze()
            

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
                    str(test_year)))
                continue

            if self.export_all_years:
                path_model = '{}/all_years/optimized_model_{}'.format(
                    self.outpath, test_year)
                path_hyperparas = '{}/all_years/hyperparameters_{}'.format(
                    self.outpath, test_year)
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
                self.best_year = test_year
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
