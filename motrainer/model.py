"""Implementing different types of neural network.

This script is for the implementation of different types of neural network,
including different structures, different loss functions.
"""

import os

import tensorflow as tf

# Force tensorflow debug logging off, keep only error logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def keras_dnn(dimensions, input_shape, output_shape):
    """Deep Neural Network implemented by Keras.

    by default:
    dimension consists of:
    learning_rate, num_dense_layers,num_input_nodes,
                 num_dense_nodes, activation
    dimension['input_shape'] = train_input.shape[1]
    dimension['output_shape'] = train_output.shape[1]
    """
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(dimensions['num_input_nodes'],
                              input_shape=(input_shape, ),
                              activation=dimensions['activation']))

    for i in range(dimensions['num_dense_layers']):
        name = f'layer_dense_{i + 1}'
        model.add(
            tf.keras.layers.Dense(dimensions['num_dense_nodes'],
                                  activation=dimensions['activation'],
                                  name=name))
    model.add(tf.keras.layers.Dense(units=output_shape))
    adam = tf.keras.optimizers.Adam(learning_rate=dimensions['learning_rate'])
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'])

    return model


def keras_dnn_lossweight(dimensions, input_shape, output_shape, loss_weights):
    """Deep Neural Network implemented by Keras.

    Implemented to adapt 'loss_weights'.
    """
    inputs = tf.keras.Input(shape=(input_shape, ))

    for i in range(dimensions['num_dense_layers']):
        name = f'layer_dense_{i + 1}'
        if i == 0:
            hidden = tf.keras.layers.Dense(dimensions['num_input_nodes'],
                                           activation=dimensions['activation'],
                                           name=name)(inputs)
            hidden_prev = hidden
        else:
            hidden = tf.keras.layers.Dense(dimensions['num_dense_nodes'],
                                           activation=dimensions['activation'],
                                           name=name)(hidden_prev)
            hidden_prev = hidden

    outputs = []
    for i in range(output_shape):
        name = f'out{i + 1}'
        outputs.append(tf.keras.layers.Dense(1, name=name)(hidden))

    adam = tf.keras.optimizers.Adam(learning_rate=dimensions['learning_rate'])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'],
                  loss_weights=loss_weights)

    return model
