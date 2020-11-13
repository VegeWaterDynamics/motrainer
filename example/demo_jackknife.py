import pickle
import pandas as pd
import logging
import sys
from ml_lsmodel_ascat.dnn import NNTrain
from ml_lsmodel_ascat.jackknife import JackknifeGPI
from ml_lsmodel_ascat.util import shap_values

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    stream=sys.stdout)

if __name__ == "__main__":
    # Manual input
    val_split_year = 2017
    out_path = './results/'
    file_data = './example_data/input_SURFEX_label_ASCAT_5GPI_2007_2019'
    list_gpi = range(5)
    output_list = ['sig', 'slop', 'curv']
    input_list = [
        'TG1', 'TG2', 'TG3', 'WG1', 'WG2', 'WG3', 'BIOMA1', 'BIOMA2',
        'RN_ISBA', 'H_ISBA', 'LE_ISBA', 'GFLUX_ISBA', 'EVAP_ISBA', 'GPP_ISBA',
        'R_ECO_ISBA', 'LAI_ISBA', 'XRS_ISBA'
    ]
    searching_space = {
        'num_dense_layers': [1, 10],
        'num_input_nodes': [1, 6],
        'num_dense_nodes': [1, 128],
        'learning_rate': [5e-4, 1e-2],
        'activation': ['relu']
    }

    optimize_space = {
        'best_loss': 1,
        'n_calls': 15,
        'epochs': 300,
        'noise': 0.01,
        'n_jobs': -1,
        'kappa': 5,
        'validation_split': 0.2,
        'x0': [1e-3, 1, 4, 13, 'relu', 64]
    }

    # read the whole data
    with open(file_data, 'rb') as f:
        clusters = pickle.load(f)
    df_all_gpi = clusters

    aprior, post = [], []
    # Loop all gpi
    for gpi_num in list_gpi:

        gpi_data = df_all_gpi.iloc[gpi_num]['data'].copy()
        gpi_data = gpi_data.dropna()

        if len(df_all_gpi) > 0:
            gpi = JackknifeGPI(gpi_data,
                               val_split_year,
                               input_list,
                               output_list,
                               outpath='{}/gpi{}'.format(out_path, gpi_num))

            gpi.train(searching_space=searching_space,
                      optimize_space=optimize_space,
                      normalize_method='standard',
                      training_method='dnn',
                      performance_method='rmse',
                      verbose=2)

            gpi.export_best()

            # Compute shap
            shaps = shap_values(gpi.best_train.model,
                                gpi.gpi_data[gpi.input_list].values)

            # Export apriori performance
            path_apriori_performance = '{}/apriori_performance_{}'.format(
                gpi.outpath, gpi.best_year)
            with open(path_apriori_performance, 'wb') as f:
                pickle.dump(gpi.apr_perf, f)

            # Export postpriori performance
            path_postpriori_performance = '{}/postpriori_performance_{}'.format(
                gpi.outpath, gpi.best_year)
            with open(path_postpriori_performance, 'wb') as f:
                pickle.dump(gpi.post_perf, f)

            # Export shap
            path_shap = '{}/shap_values_{}'.format(gpi.outpath, gpi.best_year)
            df_shap = pd.DataFrame()
            df_shap['shaps'] = shaps
            with open(path_shap, 'wb') as f:
                pickle.dump(shaps, f)

            aprior.append(gpi.apr_perf)
            post.append(gpi.post_perf)
            print("=========================================")
            print("       GPI " + str(gpi_num) + " done")
            print("       aprior performance(RMSE): ")
            print(gpi.apr_perf)
            print("=========================================")
            print("       post performance(RMSE): ")
            print(gpi.post_perf)
            print("=========================================")
        else:
            print("=========================================")
            print("       GPI" + str(gpi_num) + " is empty")
            print("=========================================")
