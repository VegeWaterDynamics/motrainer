"""
This script is for the implementation of different types of neural network,
including different structures, different loss functions
"""

import keras
import numpy as np
from pathlib import Path
from skopt.space import Real, Categorical, Integer


def keras_dnn(dimensions, input_shape, output_shape):
    """
    Deep Neural Network implemented by Keras
    by default:
    dimension consists of:
    learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes, activation
    dimension['input_shape'] = train_input.shape[1]
    dimension['output_shape'] = train_output.shape[1]
    """
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(dimensions['num_input_nodes'],
                           input_shape=(input_shape, ),
                           activation=dimensions['activation']))

    for i in range(dimensions['num_dense_layers']):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(
            keras.layers.Dense(dimensions['num_input_nodes'],
                               activation=dimensions['activation'],
                               name=name))
        model.add(keras.layers.Dense(units=output_shape))
        adam = keras.optimizers.Adam(lr=dimensions['learning_rate'])
    model.compile(optimizer=adam,
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'])

    return model
