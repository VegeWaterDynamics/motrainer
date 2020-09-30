from ml_lsmodel_ascat.dnn import TrainDNN
import pickle
import sys

if __name__ == "__main__":
    # Data Loading
    out_path ='./Jackknife_DNN_OUT/'
    var_path ='./inputs/'
    gpi_id = int(sys.argv[1])

    # Execute tranning
    tranning = TrainDNN(out_path,  var_path, gpi_id)
    tranning.run()

