from importlib_metadata import metadata
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from motrainer.dnn import NNTrain
from motrainer.util import performance, normalize

logger = logging.getLogger(__name__)


class JackknifeGPI(object):
    """
    GPI object oriented for neuron netowork training using Jackknife resampling
    method.

    Methods
    -------
    train(searching_space, optimize_space, normalize_method='standard',
          performance_method='rmse', training_method='dnn', verbose=0)
        train neuron network with given method
    export_best
        export the best results in Jackknife process.
    """

    def __init__(self,
                 gpi_data,
                 val_split_year,
                 input_list,
                 output_list,
                 export_all_years=True,
                 outpath='./jackknife_results'):
        """
        Initialize JackknifeGPI object.

        Parameters
        ----------
        gpi_data : pandas.DataFrame
            DataFrame of a single GPI.
            Each row represents all properties at a certain timestamp.
            Each column represents a time-series of a property.
        val_split_year : int
            Split year of validation. All data after (include) this year will
            be reserved for benchmarking.
        input_list : list of str
            Column names in gpi_data will will be used as input.
        output_list : list of str
            Column names in gpi_data will will be used as output.
        export_all_years : bool, optional
            Switch to export the results of all years, by default True
        outpath : str, optional
            Results exporting path, by default './jackknife_results'
        """
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

    def train(self,
              searching_space,
              optimize_space,
              normalize_method='standard',
              performance_method='rmse',
              training_method='dnn',
              verbose=0):
        """
        Train neuron network with Jackknife resampling method.
        Procedures:
        1. Reserve in/output after self.val_split_year for later benchmarking.
        2. From the rest in/output data, leave out one year as validation data.
        3. Perform neuron network training.
        4. Repeat Step 2 and 3 until all years exept benchmarking years have
            been used for validation.
        5. Select the best trainning by best performance.
        6. Perform benchmarking on reserved data.

        Parameters
        ----------
        searching_space : dict
            Arguments of searching space.
        optimize_space : dict
            Arguments of optimazation space.
        normalize_method : str, optional
            Method of normalization. Choose from 'standard' and 'min_max'.
            By default 'standard'
        performance_method : str, optional
            Method of computing performance. Choose from 'rmse', 'mae',
            'pearson' and 'spearman'.
            By default 'rmse'.
        training_method : str, optional
            Traning method selection. Select from 'dnn' or 'dnn_lossweights'.
            By default 'dnn'
        verbose : int, optional
            Control the verbosity.
            By default 0, which means no screen feedback.
        """

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

        # Jackknife in time
        loo = LeaveOneOut()
        best_perf_sum = None
        for train_index, test_index in loo.split(year_list):
            this_year = test_index[0] + year_list[0]

            logger.info('Jackknife on year: {}.'.format(str(this_year)))
            train_input = jackknife_input[
                jackknife_input.index.year != this_year]
            train_output = jackknife_output[
                jackknife_output.index.year != this_year]
            test_input = jackknife_input[jackknife_input.index.year ==
                                         this_year]
            test_output = jackknife_output[jackknife_output.index.year ==
                                           this_year]

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

    def export_best(self, model_name='best_optimized_model',
                    hyper_name='best_hyperparameters'):
        """
        export the best results in Jackknife process.
        """
        logger.info(
            'Exporting model and hyperparameters of year {} to {}'.format(
                self.best_year, self.outpath))

        if model_name is not None:
            path_model = '{}/{}_{}'.format(
                self.outpath, model_name, self.best_year)
        else:
            path_model = '{}/best_optimized_model_{}'.format(
                self.outpath, self.best_year)

        if hyper_name is not None:
            path_hyperparameters = '{}/{}_{}'.format(
                self.outpath, hyper_name, self.best_year)
        else:
            path_hyperparameters = '{}/best_hyperparameters_{}'.format(
                self.outpath, self.best_year)

        self.best_train.export(path_model=path_model,
                               path_hyperparameters=path_hyperparameters)

        # write metadata
        f_metadata = '{}/metadata.json'.format(self.outpath)
        metedata = dict()
        metedata['input_list'] = self.input_list
        metedata['output_list'] = self.input_list
        metedata['best_year'] = int(self.best_year)
        metedata['lat'] = float(self.gpi_data['lat'].iloc[0])
        metedata['lon'] = float(self.gpi_data['lon'].iloc[0])
        with open(f_metadata, 'w') as f:
            json.dump(metedata, f)