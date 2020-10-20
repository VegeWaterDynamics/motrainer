import pickle
from ml_lsmodel_ascat.dnn import NNTrain
from ml_lsmodel_ascat.jackknife import JackknifeGPI
from ml_lsmodel_ascat.util import shap_values

if __name__ == "__main__":
    # Manual input
    val_split_year = 2017
    out_path = './results/'
    file_data = './example_data/input_SURFEX_label_ASCAT_3GPI_2007_2019'
    list_gpi = [3]
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
    df_all_gpi = clusters

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

            gpi.train(searching_space, optimize_space, 'standard', 'rmse',
                      val_split_year)

            gpi.export_best()

            # Compute shap
            shap_values = shap_values(gpi.best_train.model,
                                      gpi.gpi_input.values)

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
            with open(path_shap, 'wb') as f:
                pickle.dump([
                    zip(
                        shap_values[0][:, 0],
                        shap_values[0][:, 1],
                        shap_values[0][:, 2],
                        shap_values[0][:, 3],
                        shap_values[1][:, 0],
                        shap_values[1][:, 1],
                        shap_values[1][:, 2],
                        shap_values[1][:, 3],
                        shap_values[2][:, 0],
                        shap_values[2][:, 1],
                        shap_values[2][:, 2],
                        shap_values[2][:, 3],
                    )
                ], f)

            print("=========================================")
            print("       GPI " + str(gpi_num) + " done")
            print("=========================================")
        else:
            print("=========================================")
            print("       GPI" + str(gpi_num) + " is empty")
            print("=========================================")
