import numpy as np
import shap
import sklearn
from scipy.stats.stats import pearsonr, spearmanr


def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                              1000,
                                              replace=False)]

    shap.initjs()
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)

    return shap_values


def performance(data_input, data_label, model, method, scaler_output=None):
    predicted = model.predict(data_input)

    # Scale back if the data was normalized
    if scaler_output is not None:
        re_predicted = scaler_output.inverse_transform(predicted, 'f')
        re_label = scaler_output.inverse_transform(data_label, 'f')
    else:
        re_predicted = predicted
        re_label = data_label

    difference = re_predicted - re_label
    perf = np.zeros([predicted.shape[1], 1])
    if method == 'rmse':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(np.sqrt(((difference[j])**2).mean()), 5)
    elif method == 'mae':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round((difference[j].mean()), 5)
    elif method == 'pearson':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(pearsonr(re_predicted[j], re_label[j]), 5)[0]
    elif method == 'spearman':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(spearmanr(re_predicted[j], re_label[j]),
                                  5)[0]

    return perf


def normalize(data, method):
    # prenormalization for output (or label)
    # we just need do normalization for output and input seperately
    # which means mean/std are same for train and test
    if method == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif method == 'min_max':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        raise NotImplementedError

    data_norm = scaler.fit_transform(data)
    return data_norm, scaler