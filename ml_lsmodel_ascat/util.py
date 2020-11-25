import numpy as np
import tensorflow as tf
import os
import shap
import sklearn
import random
from scipy.stats.stats import pearsonr, spearmanr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Force tensorflow debug logging off


def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                              1000,
                                              replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)

    return shap_values


def performance(data_input, data_label, model, method, scaler_output=None):
    # Temporally SL the model because of TF graph execution issue
    # TODO: fix the model prediction issue
    tmp_path = '/tmp/tmp_model{}'.format(random.getrandbits(64))
    model.save(tmp_path)
    model = tf.keras.models.load_model(tmp_path)
    predicted = model.predict(data_input)
    
    # In case multiple outputs, re-arrange to one df
    if isinstance(predicted, list):
        predicted = np.hstack(predicted)

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
