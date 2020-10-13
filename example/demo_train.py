from ml_lsmodel_ascat.dnn import DNNTrain
import pickle
import sys

if __name__ == "__main__":
    # Data Loading
    out_path ='./results/'
    file_data ='/mnt/c/Users/OuKu/Developments/Global_vegetation/data/SURFEX/input_SURFEX_label_ASCAT_9GPI_2007_2019'
    
    with open(file_data, 'rb') as f:
        clusters = pickle.load(f)
    gpi_data = clusters.iloc[5].data
    gpi_input = gpi_data[['TG1', 'TG2', 'TG3', 'WG1', 'WG2', 'WG3', 
                           'BIOMA1', 'BIOMA2', 'RN_ISBA', 'H_ISBA', 
                           'LE_ISBA', 'GFLUX_ISBA', 'EVAP_ISBA', 
                           'GPP_ISBA', 'R_ECO_ISBA', 'LAI_ISBA', 'XRS_ISBA']]
    gpi_output = gpi_data[['sig', 'slop']]                      
    del clusters
    train_input = gpi_input[gpi_input.index.year<2017].values
    train_output = gpi_output[gpi_output.index.year<2017].values
    test_input = gpi_input[gpi_input.index.year==2017].values
    test_output = gpi_output[gpi_output.index.year==2017].values
    vali_input =  gpi_input[gpi_input.index.year>2017].values
    vali_output = gpi_output[gpi_output.index.year>2017].values

    # Execute tranning
    tranning = DNNTrain(train_input, train_output,
                        test_input, test_output,
                        vali_input, vali_output,
                        out_path)

    tranning.update_paras(learning_rate = [5e-4, 1e-2],
                          activation = ['relu'])

    tranning.normalize()

    tranning.optimize(best_loss=5)

    
    pass

