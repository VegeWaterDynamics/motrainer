#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 22:42:11 2020

@author: xushan
"""
import pickle
from ml_lsmodel_ascat.dnn import DNNTrain
from ml_lsmodel_ascat.gpi_jackknife import JackknifeGPI

if __name__ == "__main__":
    # Manual input
    val_split_year = 2017
    out_path = './results/'
    file_data = '/mnt/c/Users/OuKu/Developments/Global_vegetation/data/SURFEX/input_SURFEX_label_ASCAT_9GPI_2007_2019'
    list_gpi = [5]
    output_list = ['sig', 'slop', 'curv']
    input_list = [
        'TG1', 'TG2', 'TG3', 'WG1', 'WG2', 'WG3', 'BIOMA1', 'BIOMA2',
        'RN_ISBA', 'H_ISBA', 'LE_ISBA', 'GFLUX_ISBA', 'EVAP_ISBA', 'GPP_ISBA',
        'R_ECO_ISBA', 'LAI_ISBA', 'XRS_ISBA'
    ]
    searching_space = {'learning_rate': [5e-4, 1e-2], 'activation': ['relu']}
    optimize_space = {
        'best_loss': 1,
        'n_calls': 15,
        'noise': 0.01,
        'n_jobs': -1,
        'kappa': 5,
        'validation_split': 0.2,
        'x0': [1e-3, 1, 4, 13, 'relu', 64]
    }

    # read the whole data
    with open(file_data, 'rb') as f:
        clusters = pickle.load(f)
    df_all_gpi = clusters  #.iloc[gpi_num].data

    # Loop all gpi
    for gpi_num in list_gpi:
        gpi_data = df_all_gpi.iloc[gpi_num]['data']
        gpi_data = gpi_data.dropna()

        if len(df_all_gpi) > 0:
            gpi = JackknifeGPI(gpi_data,
                               val_split_year,
                               input_list,
                               output_list,
                               outpath='{}/gpi{}'.format(out_path, gpi_num))

            gpi.normalize(norm_method='standard')

            gpi.train_test(searching_space, optimize_space, 'rmse',
                           val_split_year)

            gpi.export_best()

            print("=========================================")
            print("       GPI " + str(gpi_num) + " done")
            print("=========================================")
        else:
            print("=========================================")
            print("       GPI" + str(gpi_num) + " is empty")
            print("=========================================")
