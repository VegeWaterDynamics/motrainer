import base64
import os
import pickle
import random

import h5py
import numpy as np
import sklearn.preprocessing
from scipy.stats import pearsonr, spearmanr

# Force tensorflow debug logging off, keep only error logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # noqa: E402


def performance(data_input, data_label, model, method, scaler_output=None):
    """Compute performance of trained neuron netowrk.

    Parameters
    ----------
    data_input : pandas.DataFrame
        Input data.
    data_label : pandas.DataFrame
        Label data.
    model : tf.keras.models
        Trained model to compute performance.
    method : str
        Method to compute
    scaler_output : optional
        Scaler of output, by default None.
        When not None, function will assume that a normalization has been
        performed to output, and will use scaler_output to transform the output
        back to the original scale.

    Returns
    -------
    float or list of float
        Performance value. If the model gives multiple output, the performance
        will be a list.
    """
    # Temporally SL the model because of TF graph execution issue
    # TODO: fix the model prediction issue
    tmp_path = f'/tmp/tmp_model{random.getrandbits(64)}'
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
            perf[j, 0] = np.round(pearsonr(re_predicted[:, j], re_label[:, j]),
                                  5)[0]
    elif method == 'spearman':
        for j in range(predicted.shape[1]):
            perf[j,
                 0] = np.round(spearmanr(re_predicted[:, j], re_label[:, j]),
                               5)[0]

    return perf


def normalize(data, method):
    """Pre-normalization for input/output.

    Parameters
    ----------
    data : pandas.DataFrAME
        Data to normalize.
    method : str
        Data to normalize. Choose from 'standard' or 'min_max'.

    Returns
    -------
    list
        A list of [data_norm, scaler]. Normalized data and scaler used for
        normalization.

    """
    if method == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif method == 'min_max':
        scaler = sklearn.preprocessing.MinMaxScaler()
    else:
        raise NotImplementedError

    data_norm = scaler.fit_transform(data)
    return data_norm, scaler


def sklearn_save(model, path_model, meta_data=None):
    """Save sklearn model to hdf5 file.

    Parameters
    ----------
    model : sklearn.model
        Sklearn model to save.
    path_model : str
        Path to save the model.
    meta_data : Dict, optional
        optional. A dict of meta data to save.

    """
    model_bytes = pickle.dumps(model)

    # Encode the bytes as base64
    model_base64 = base64.b64encode(model_bytes)

    with h5py.File(path_model, 'w') as f:
        f.attrs['model'] = model_base64

        if meta_data is not None:
            for key, value in meta_data.items():
                f.attrs[key] = value


def sklearn_load(path_model):
    """Load sklearn model from hdf5 file.

    Parameters
    ----------
    path_model : str
        Path to the model.

    Returns
    -------
    sklearn.model
        Sklearn model.

    """
    with h5py.File(path_model, 'r') as f:
        if 'model' not in f.attrs:
            raise ValueError("No model found in the hdf5 file.")

        model_base64 = f.attrs['model']

        # Decode the bytes
        model_bytes = base64.b64decode(model_base64)

        # Load the model
        model = pickle.loads(model_bytes)

        meta_data = {}
        for key in f.attrs.keys():
            if key != 'model':
                meta_data[key] = f.attrs[key]


    return model, meta_data
