import numpy as np
import tensorflow as tf
import os
import shap
import sklearn
import random
from scipy.stats.stats import pearsonr, spearmanr
from shapely.geometry import Point

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Force tensorflow debug logging off


def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                              1000,
                                              replace=False)]

    # concatenate multiple outputs for shap calculation
    if len(model.outputs) > 1:
        model = tf.keras.Model(inputs=model.inputs,
                               outputs=tf.keras.layers.concatenate(
                                   model.outputs))

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
            perf[j, 0] = np.round(pearsonr(re_predicted[:, j], re_label[:, j]),
                                  5)[0]
    elif method == 'spearman':
        for j in range(predicted.shape[1]):
            perf[j,
                 0] = np.round(spearmanr(re_predicted[:, j], re_label[:, j]),
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


def geom_to_masked_cube(df, geometry, lats, lons, mask_excludes=True):
    # Get horizontal coords for masking purposes.

    mask_t = []
    # Iterate through all horizontal points in cube, and
    # check for containment within the specified geometry.
    for lat, lon in zip(lats, lons):
        this_point = Point(lon, lat)
        res = geometry.contains(this_point)
        mask_t.append(res.values[0])

    mask_t = np.array(mask_t)
    if mask_excludes:
        # Invert the mask if we want to include the geometry's area.
        mask_t = ~mask_t

    # Make sure the mask is the same shape as the cube.
    # Apply the mask to the cube's data.
    df_copy = df.copy()
    data = df_copy.values
    masked_data = np.ma.masked_array(data, mask_t)
    df_copy = masked_data
    return df_copy
