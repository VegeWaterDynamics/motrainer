#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:49:32 2020

@author: xushan x.shan-2@tudelft.nl
"""
import numpy as np
from pathlib import Path
from ml_lsmodel_ascat import dnn
from ml_lsmodel_ascat.dnn import DNNTrain
#from dnn import DNNTrain
import pickle
import sklearn
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneOut
from keras.models import load_model


#if __name__ == "__main__":
class single_gpi(object):
    def __init__(self, 
                 gpi_data, gpi_num, val_split_year,
                 input_list, output_list,
                 out_path='.'):
        self.gpi_data    = gpi_data.dropna()
        self.gpi_num     = gpi_num
        self.input_list  = input_list
        self.output_list = output_list
        self.gpi_input   = gpi_data[input_list].dropna()
        self.gpi_output  = gpi_data[output_list].dropna()
        self.out_path    = Path(out_path)
        self.val_split_year=val_split_year
    
    def normalize(self, norm_method):
        # prenormalization for output (or label)
        # we just need do normalization for output and input seperately
        # which means mean/std are same for train and test
        if norm_method == 'standard':
            self.scaler_output = sklearn.preprocessing.StandardScaler()
            self.scaler_input  = sklearn.preprocessing.StandardScaler()
        elif norm_method == 'min_max':
            self.scaler_output = sklearn.preprocessing.MinMaxScaler()
            self.scaler_input  = sklearn.preprocessing.MinMaxScaler()
        else:
            print("Incorrect input strings for normalization methods")
#            break

        self.gpi_data[self.input_list] = self.scaler_input.fit_transform(self.gpi_data[self.input_list])
        self.gpi_data[self.output_list] = self.scaler_output.fit_transform(self.gpi_data[self.output_list])
        
        
    def train_test(self, 
                   searching_space, optimize_space,
                   performance_method='rmse', val_split_year=2017):
        
#        self.searching_space = searching_space
#        self.optimize_space  = optimize_space
#        gpi_data = gpi_data.dropna()
        
        # input
#        var_list = list(gpi_data.columns)
#        output_list = ['sig', 'slop', 'curv']
#        input_list  = ['TG1', 'TG2', 'TG3', 'WG1', 'WG2', 'WG3', 
#                       'BIOMA1', 'BIOMA2', 'RN_ISBA', 'H_ISBA', 
#                       'LE_ISBA', 'GFLUX_ISBA', 'EVAP_ISBA',
#                       'GPP_ISBA', 'R_ECO_ISBA', 'LAI_ISBA', 'XRS_ISBA']
        
#        gpi_input = gpi_data[input_list]
#        gpi_output = gpi_data[output_list]                      
        
        df_all_gpi_norm = self.gpi_data.dropna()
        
        jackknife_all = df_all_gpi_norm[df_all_gpi_norm.index.year < self.val_split_year]
        self.vali_all = df_all_gpi_norm[df_all_gpi_norm.index.year >= self.val_split_year]
        
        year_list = jackknife_all.copy().resample('Y').mean().index.year
        
        self.jackknife_metrics = np.zeros([len(self.output_list),len(year_list)]) 
        
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(year_list):
            
            train_all = jackknife_all[(jackknife_all.index.year != test_index[0]+year_list[0])]
            test_all  = jackknife_all[(jackknife_all.index.year == test_index[0]+year_list[0])]
            train_input, train_output = train_all[self.input_list].values, train_all[self.output_list].values
            test_input,  test_output  = test_all[self.input_list].values,  test_all[self.output_list].values
            
            # Execute training
            training = DNNTrain(train_input, train_output,
                                test_input, test_output,
#                                vali_input, vali_output,
                                self.out_path)
            
            # Normalize data
    #        training.normalize()
            # notice, normalize should be done to the whole time series!!
    
            # Set searching space
            training.update_space(learning_rate = [searching_space['learning_rate'][0],
                                                   searching_space['learning_rate'][1]],
                                  activation = searching_space['activation'])
    
            # Optimization
            training.optimize(best_loss=optimize_space['best_loss'],
                              n_calls=optimize_space['n_calls'],
                              noise= optimize_space['noise'],
                              n_jobs=optimize_space['n_jobs'],
                              kappa = optimize_space['kappa'],
                              validation_method = optimize_space['validation_method'],
                              x0=optimize_space['x0'])
            
            # Export results
            training.export_model('gpi'+str(self.gpi_num), str(test_index[0]+year_list[0]))
            model = training.get_model('gpi'+str(self.gpi_num), str(test_index[0]+year_list[0]))
            predicted = model.predict(training.test_input)
            
            print('=====================================')
            print('jackknife on '+str(test_index[0]+2010))
            print('=====================================')
            
            # calculate the metrics
            # rmse, mae, pearso, spearson
            temp = training.performance(self.scaler_output,'rmse', predicted)
            self.jackknife_metrics[:,test_index[0]]=temp.reshape(len(self.output_list),)
            
        dnn.save_performance(self.out_path, self.jackknife_metrics, 'rmse',
                             'gpi'+str(self.gpi_num))
        
        rmse_array = np.nansum(self.jackknife_metrics, axis = 0)#[0,:]
        self.rmse_min_year = np.nanargmin(rmse_array)+year_list[0]   
        self.best_model = load_model('{}/{}/optimized_model_{}'.format(self.out_path.as_posix(), 
                                     'gpi'+str(self.gpi_num), str(self.rmse_min_year)))
        self.best_model.save('{}/{}/best_optimized_model_{}'.format(self.out_path.as_posix(), 
                             'gpi'+str(self.gpi_num), str(self.rmse_min_year)))
        
        df = open('{}/{}/Hyperparameter_space_{}'.format(self.out_path.as_posix(),
                  'gpi'+str(self.gpi_num), str(self.rmse_min_year)), 'rb')
        best_hyper = pickle.load(df)
        df.close()
        best_hyper = best_hyper[0]
        
        with open('{}/{}/Best_Hyperpara_space_{}'.format(self.out_path.as_posix(),
                  'gpi'+str(self.gpi_num), str(self.rmse_min_year)), 'wb') as f:
            pickle.dump(best_hyper, f)
        
        # evaluate the model using validation data
        
    def shap_values(self):
        # calculate the shap value
        input_whole = self.gpi_input.values
        self.shap_values = dnn.get_shap_values(self.best_model, input_whole)
        dnn.export_shap_values(self.out_path.as_posix(), self.shap_values,
                               'gpi'+str(self.gpi_num), str(self.rmse_min_year))
    
    def val_best_model(self, method):
        vali_input  = self.vali_all[self.input_list].values
        vali_output = self.vali_all[self.output_list].values
        
        predicted = self.best_model.predict(vali_input)
        
        re_predicted = self.scaler_output.inverse_transform(predicted,'f')
        re_label     = self.scaler_output.inverse_transform(vali_output,'f')
        
        difference = re_predicted - re_label
        self.performance = np.zeros([predicted.shape[1],1])
        if method == 'rmse':
            for j in range(predicted.shape[1]):
                self.performance[j,0] = np.round(np.sqrt(((difference[j]) ** 2).mean()),5)
        elif method == 'mae':
            for j in range(predicted.shape[1]):
                self.performance[j,0] = np.round((difference[j].mean()),5)
        elif method == 'pearson':
            for j in range(predicted.shape[1]):
                self.performance[j,0] = np.round(pearsonr(re_predicted[j], re_label[j]),5)[0]
        elif method == 'spearman':
            for j in range(predicted.shape[1]):
                self.performance[j,0] = np.round(spearmanr(re_predicted[j], re_label[j]),5)[0]
        
    def save_val_performance(self):
        with open('{}/{}/Val_performance'.format(self.out_path.as_posix(),
                  'gpi'+str(self.gpi_num)), 'wb') as f:
            pickle.dump(self.performance, f)
#        return self.performance
        
