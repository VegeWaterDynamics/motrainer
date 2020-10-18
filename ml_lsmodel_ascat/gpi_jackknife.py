import numpy as np
import pickle
import sklearn
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from ml_lsmodel_ascat.dnn import DNNTrain
from ml_lsmodel_ascat.util import shap_values


class JackknifeGPI(object):
    def __init__(self,
                 gpi_data,
                 val_split_year,
                 input_list,
                 output_list,
                 export_all_years=True,
                 outpath='./jackknife_results'):
        self.gpi_data = gpi_data.dropna()
        self.input_list = input_list
        self.output_list = output_list
        self.gpi_input = gpi_data[input_list].dropna()
        self.gpi_output = gpi_data[output_list].dropna()
        self.val_split_year = val_split_year
        self.export_all_years = export_all_years
        self.outpath = outpath

    def normalize(self, norm_method):
        # prenormalization for output (or label)
        # we just need do normalization for output and input seperately
        # which means mean/std are same for train and test
        if norm_method == 'standard':
            self.scaler_output = sklearn.preprocessing.StandardScaler()
            self.scaler_input = sklearn.preprocessing.StandardScaler()
        elif norm_method == 'min_max':
            self.scaler_output = sklearn.preprocessing.MinMaxScaler()
            self.scaler_input = sklearn.preprocessing.MinMaxScaler()
        else:
            print("Incorrect input strings for normalization methods")

        self.gpi_data[self.input_list] = self.scaler_input.fit_transform(
            self.gpi_data[self.input_list])
        self.gpi_data[self.output_list] = self.scaler_output.fit_transform(
            self.gpi_data[self.output_list])

    def train_test(self,
                   searching_space,
                   optimize_space,
                   performance_method='rmse',
                   val_split_year=2017):
                           
        df_all_gpi_norm = self.gpi_data.dropna()

        jackknife_all = df_all_gpi_norm[
            df_all_gpi_norm.index.year < self.val_split_year]

        year_list = jackknife_all.copy().resample('Y').mean().index.year

        loo = LeaveOneOut()

        best_performance = None
        for train_index, test_index in loo.split(year_list):
            this_year = test_index[0] + year_list[0]

                    
            print('=====================================')
            print('jackknife on '+str(this_year))
            print('=====================================')

            train_all = jackknife_all[(jackknife_all.index.year != this_year)]
            test_all = jackknife_all[(jackknife_all.index.year == this_year)]
            train_input, train_output = train_all[
                self.input_list].values, train_all[self.output_list].values
            test_input, test_output = test_all[
                self.input_list].values, test_all[self.output_list].values

            # Execute training
            training = DNNTrain(train_input, train_output, test_input,
                                test_output)

            # Set searching space
            training.update_space(learning_rate=[
                searching_space['learning_rate'][0],
                searching_space['learning_rate'][1]
            ],
                                  activation=searching_space['activation'])

            # Optimization
            training.optimize(
                best_loss=optimize_space['best_loss'],
                n_calls=optimize_space['n_calls'],
                noise=optimize_space['noise'],
                n_jobs=optimize_space['n_jobs'],
                kappa=optimize_space['kappa'],
                validation_method=optimize_space['validation_method'],
                x0=optimize_space['x0'])

            training.get_performance(self.scaler_output, 'rmse')

            # find minimum rmse
            # TODO: mae, pearson, spearman
            rmse = np.nansum(training.performance)
            if best_performance is None:
                best_performance = rmse
            elif rmse < best_performance:
                self.best_performance = rmse
                self.best_train = training
                self.best_year = this_year
                self.shap_values = shap_values(self.best_train.model,
                                               self.gpi_input.values)

            if self.export_all_years:
                path_model = '{}/all_years/optimized_model_{}'.format(
                    self.outpath, this_year)
                path_hyperparameters = '{}/all_years/hyperparameters_{}'.format(
                    self.outpath, this_year)
                training.export(
                    path_model=path_model,
                    path_hyperparameters=path_hyperparameters)

    def export_best(
            self,
            output_options=['model', 'hyperparameters', 'performance',
                            'shap']):

        if 'model' in output_options:
            path_model = '{}/best_optimized_model_{}'.format(
                self.outpath, self.best_year)

        if 'hyperparameters' in output_options:
            path_hyperparameters = '{}/best_hyperparameters_{}'.format(
                self.outpath, self.best_year)

        if 'performance' in output_options:
            path_performance = '{}/best_performance_{}'.format(
                self.outpath, self.best_year)

        self.best_train.export(path_model=path_model,
                               path_hyperparameters=path_hyperparameters,
                               path_performance=path_performance)

        if 'shap' in output_options:
            path_shap = '{}/shap_values_{}'.format(self.outpath,
                                                   self.best_year)
            Path(path_shap).parent.mkdir(parents=True, exist_ok=True)
            with open(path_shap, 'wb') as f:
                pickle.dump([
                    zip(
                        self.shap_values[0][:, 0],
                        self.shap_values[0][:, 1],
                        self.shap_values[0][:, 2],
                        self.shap_values[0][:, 3],
                        self.shap_values[1][:, 0],
                        self.shap_values[1][:, 1],
                        self.shap_values[1][:, 2],
                        self.shap_values[1][:, 3],
                        self.shap_values[2][:, 0],
                        self.shap_values[2][:, 1],
                        self.shap_values[2][:, 2],
                        self.shap_values[2][:, 3],
                    )
                ], f)
