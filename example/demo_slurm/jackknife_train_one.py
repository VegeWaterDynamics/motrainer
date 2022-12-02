import pickle
import sys
from motrainer.jackknife import JackknifeGPI

if __name__ == "__main__":
    # Parsing input
    gpi_id = int(sys.argv[1])

    # Manual input
    out_path = './results/'
    file_data = '../example_data/example_data.pickle'
    val_split_year = 2017
    output_list = ['sig', 'slop', 'curv']
    input_list = [
        'TG1', 'TG2', 'TG3', 'WG1', 'WG2', 'WG3', 'BIOMA1', 'BIOMA2']
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

    # Read example data
    df_all_gpi = pd.read_pickle(file_data)

    gpi_data = df_all_gpi.iloc[gpi_id]['data']
    gpi_data = gpi_data.dropna()

    if len(gpi_data) > 0:
        gpi = JackknifeGPI(gpi_data,
                           val_split_year,
                           input_list,
                           output_list,
                           outpath='{}/gpi{}'.format(out_path, gpi_id))

        gpi.train(searching_space=searching_space,
                  optimize_space=optimize_space,
                  normalize_method='standard',
                  training_method='dnn',
                  performance_method='rmse',
                  val_split_year=val_split_year)

        gpi.export_best()

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


        print("=========================================")
        print("       GPI " + str(gpi_id) + " done")
        print("=========================================")
    else:
        print("=========================================")
        print("       GPI" + str(gpi_id) + " is empty")
        print("=========================================")
