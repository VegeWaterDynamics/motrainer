"""
This script is for the implementation of different types of neural network,
including different structures, different loss functions
"""

import os
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
from skopt.space import Real, Categorical, Integer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Force tensorflow debug logging off


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
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(dimensions['num_input_nodes'],
                              input_shape=(input_shape, ),
                              activation=dimensions['activation']))

    for i in range(dimensions['num_dense_layers']):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(
            tf.keras.layers.Dense(dimensions['num_input_nodes'],
                                  activation=dimensions['activation'],
                                  name=name))
    model.add(tf.keras.layers.Dense(units=output_shape))
    adam = tf.keras.optimizers.Adam(lr=dimensions['learning_rate'])
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'])

    return model


def keras_dnn_lossweight(dimensions, input_shape, output_shape, loss_weights):
    """
    """
    inputs = tf.keras.Input(shape=(input_shape, ))

    for i in range(dimensions['num_dense_layers']):
        name = 'layer_dense_{0}'.format(i + 1)
        if i == 0:
            hidden = tf.keras.layers.Dense(dimensions['num_input_nodes'],
                                           activation=dimensions['activation'],
                                           name=name)(inputs)
            hidden_prev = hidden
        else:
            hidden = tf.keras.layers.Dense(dimensions['num_input_nodes'],
                                           activation=dimensions['activation'],
                                           name=name)(hidden_prev)
            hidden_prev = hidden

    outputs = []
    for i in range(output_shape):
        name = 'out{}'.format(i + 1)
        outputs.append(tf.keras.layers.Dense(1, name=name)(hidden))

    adam = tf.keras.optimizers.Adam(lr=dimensions['learning_rate'])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae', 'acc'],
                  loss_weights=loss_weights)

    return model

#def keras_lstm(learning_rate, momentum, num_filters, #num_kernel_size,
#               input_timesteps, input_features, 
#               output_feature,
#               dropout_rate,
#                 activation):
def keras_lstm(dimensions,
               input_timesteps, input_features, 
               output_feature
               ):
    sgd_method = tf.keras.optimizers.SGD(learning_rate=dimensions['learning_rate'],
                     momentum = dimensions['momentum'], nesterov=False)
    
    model = tf.keras.models.Sequential()
#    inputs = tf.keras.Input(shape=(input_shape, ))
#    model.add(tf.keras.Input(shape=(dimensions['batch_size'],input_features,input_timesteps)))
#    model.add(
#        tf.keras.layers.Dense(dimensions['num_input_nodes'],
#                              input_shape=(input_features,input_timesteps,),
#                              activation=dimensions['activation']))
    for i in range(dimensions['num_dense_layers']):
        name = 'layer_dense_{0}'.format(i)
        if i == 0:
            model.add(tf.keras.layers.LSTM(units = dimensions['num_filters'],
#                                           batch_input_shape=(
#                                                   dimensions['batch_size'],
#                                                   input_timesteps,
#                                                   input_features),
                                           input_shape=(
                                                   input_timesteps,
                                                   input_features),
                                           activation=dimensions['activation'],
                                           dropout=dimensions['dropout_rate'],
                                           return_sequences=True,
                                           name=name
                                           ))
        elif i < (dimensions['num_dense_layers']-1) and i >0:
            model.add(tf.keras.layers.LSTM(units = dimensions['num_filters'],
                                           activation=dimensions['activation'],
                                           dropout=dimensions['dropout_rate'],
                                           return_sequences=True,
                                           name=name
                                           ))
        else:
            model.add(tf.keras.layers.LSTM(units = dimensions['num_filters'],
                                           activation=dimensions['activation'],
                                           dropout=dimensions['dropout_rate'],
                                           return_sequences=False,
#                                           return_sequences=True,
                                           name=name
                                           ))
    
    
    model.add(tf.keras.layers.Dense(units = output_feature, activation='linear', name='output'))
    print(model.summary())

    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer= sgd_method,#adam, #'rmsprop',
                  metrics=['mae', 'acc'])

    return model