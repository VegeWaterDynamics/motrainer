import numpy as np
import pickle
import sklearn
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from ml_lsmodel_ascat.dnn import DNNTrain
from ml_lsmodel_ascat.util import shap_values

class single_gpi(object):
    def __init__(self,
                 gpi_data,
                 gpi_num,
                 val_split_year,
                 input_list,
                 output_list):
        self.gpi_data = gpi_data.dropna()
        self.gpi_num = gpi_num
        self.input_list = input_list
        self.output_list = output_list
        self.gpi_input = gpi_data[input_list].dropna()
        self.gpi_output = gpi_data[output_list].dropna()
        self.val_split_year = val_split_year

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
        # self.vali_all = df_all_gpi_norm[
        #     df_all_gpi_norm.index.year >= self.val_split_year]

        year_list = jackknife_all.copy().resample('Y').mean().index.year

        # self.jackknife_metrics = np.zeros(
        #     [len(self.output_list), len(year_list)])

        loo = LeaveOneOut()

        min_rmse = None
        for train_index, test_index in loo.split(year_list):

            train_all = jackknife_all[(jackknife_all.index.year !=
                                       test_index[0] + year_list[0])]
            test_all = jackknife_all[(
                jackknife_all.index.year == test_index[0] + year_list[0])]
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
            rmse = np.nansum(training.performance)
            if min_rmse is None:
                min_rmse = rmse
            elif rmse < min_rmse:
                self.min_rmse = rmse
                self.best_train = training
                self.rmse_min_year = test_index[0] + year_list[0]
                self.shap_values = shap_values(self.best_train.model, self.gpi_input.values)

    def export_best(self, 
                    path_shap=None, 
                    **kwargs):
                
        self.best_train.export(**kwargs)

        if path_shap is not None:
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
         
