"""
This script is for the implementation of different types of neural network,
including different structures, different loss functions
"""

from tensorflow import keras
import numpy as np
from pathlib import Path
from skopt.space import Real, Categorical, Integer
from tensorflow.keras.layers import Input, Dense


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


def keras_dnn_lossweight(dimensions, input_shape, output_shape, loss_weights):
    """
    """
    inputs = Input(shape=(input_shape, ))

    for i in range(dimensions['num_dense_layers']):
        name = 'layer_dense_{0}'.format(i + 1)
        if i == 0:
            hidden = Dense(dimensions['num_input_nodes'],
                           activation=dimensions['activation'],
                           name=name)(inputs)
            hidden_prev = hidden
        else:
            hidden = Dense(dimensions['num_input_nodes'],
                           activation=dimensions['activation'],
                           name=name)(hidden_prev)
            hidden_prev = hidden

    outputs = []
    for i in range(output_shape):
        name = 'out{}'.format(i + 1)
        outputs.append(Dense(1, name=name)(hidden))

    adam = keras.optimizers.Adam(lr=dimensions['learning_rate'])

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam,
                  loss=keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'],
                  loss_weights=loss_weights)

    return model
