import logging
import tensorflow as tf
import sklearn
import skopt
import numpy as np
import pickle
from pathlib import Path
from skopt.space import Real, Categorical, Integer
from ml_lsmodel_ascat.model import keras_dnn, keras_dnn_lossweight

logger = logging.getLogger(__name__)


class NNTrain(object):
    def __init__(self, train_input, train_output):
        self.train_input = train_input
        self.train_output = train_output
        self.dimensions = {
            'learning_rate':
            Real(low=5e-4,
                 high=1e-2,
                 prior='log-uniform',
                 name='learning_rate'),
            'num_dense_layers':
            Integer(low=1, high=2, name='num_dense_layers'),
            'num_input_nodes':
            Integer(low=2, high=6, name='num_input_nodes'),
            'num_dense_nodes':
            Integer(low=1, high=128, name='num_dense_nodes'),
            'activation':
            Categorical(categories=['relu'], name='activation'),
            'batch_size':
            Integer(low=7, high=365, name='batch_size')
        }
        self.model = None

    def update_space(self, **kwrags):
        for key, value in kwrags.items():
            logger.debug('Update seaching sapce: {}={}'.format(key, value))
            # skopt.space instances
            if isinstance(value, (Real, Categorical, Integer)):
                self.dimensions[key] = value
                self.dimensions[key].name = key

            # Float searching space, e.g. learning rates
            elif any([isinstance(obj, float) for obj in value]):
                assert len(value) == 2
                if any([isinstance(obj, int) for obj in value]):
                    logger.warning(
                        'Mixed fload/int type found in {}:{}. '
                        'The search space will be interpreted as float. '
                        'If this behavior is not desired, try to specify'
                        'all elements in {} with the same type.'.format(
                            key, value, key))
                self.dimensions[key] = Real(low=value[0],
                                            high=value[1],
                                            prior='log-uniform',
                                            name=key)

            # Integer searching space, e.g. num of input nodes
            elif all([isinstance(obj, int) for obj in value]):
                assert len(value) == 2
                self.dimensions[key] = Integer(low=value[0],
                                               high=value[1],
                                               name=key)

            # Categorical searching space, e.g. activation
            elif all([isinstance(obj, str) for obj in value]):
                self.dimensions[key] = Categorical(categories=value, name=key)

            else:
                logger.error(
                    'Do not understand searching space: {}:{}.'.format(
                        key, value))
                raise NotImplementedError

    def optimize(self,
                 best_loss=1,
                 n_calls=15,
                 epochs=100,
                 noise=0.01,
                 n_jobs=-1,
                 kappa=5,
                 validation_split=0.2,
                 x0=[1e-3, 1, 4, 13, 'relu', 64],
                 training_method='dnn',
                 loss_weights=None,
                 verbose=0):
        self.best_loss = best_loss
        self.keras_verbose = verbose
        self.loss_weights = loss_weights

        @skopt.utils.use_named_args(dimensions=list(self.dimensions.values()))
        def func(**dimensions):
            logger.info('optimizing with dimensions: {}'.format(dimensions))

            # setup model
            earlystop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                verbose=self.keras_verbose,
                patience=30)

            if training_method == 'dnn':
                model = keras_dnn(dimensions, self.train_input.shape[1],
                                  self.train_output.shape[1])
                train_output = self.train_output
            elif training_method == 'dnn_lossweights':
                if self.loss_weights is None:
                    self.loss_weights = [1] * self.train_output.shape[1]
                    logger.warning('loss_weights is None.'
                                   'Using default weights {}'.format(
                                       self.loss_weights))
                model = keras_dnn_lossweight(dimensions,
                                             self.train_input.shape[1],
                                             self.train_output.shape[1],
                                             self.loss_weights)
                train_output = [
                    self.train_output.iloc[:, i]
                    for i in range(self.train_output.shape[1])
                ]

            # Fit model
            blackbox = model.fit(x=self.train_input,
                                 y=train_output,
                                 epochs=epochs,
                                 batch_size=dimensions['batch_size'],
                                 callbacks=[earlystop],
                                 verbose=self.keras_verbose,
                                 validation_split=validation_split)

            # Get loss
            loss = blackbox.history['val_loss'][-1]
            if loss < self.best_loss:
                self.model = model
                self.best_loss = loss
            del model
            tf.keras.backend.clear_session()
            return loss

        self.gp_result = skopt.gp_minimize(func=func,
                                           dimensions=list(
                                               self.dimensions.values()),
                                           n_calls=n_calls,
                                           noise=noise,
                                           n_jobs=n_jobs,
                                           kappa=kappa,
                                           x0=x0)

    def export(self, path_model=None, path_hyperparameters=None):

        if path_model is not None:
            Path(path_model).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(path_model)

        if path_hyperparameters is not None:
            Path(path_hyperparameters).parent.mkdir(parents=True,
                                                    exist_ok=True)
            with open(path_hyperparameters, 'wb') as f:
                pickle.dump([
                    sorted(
                        zip(self.gp_result.func_vals, self.gp_result.x_iters))
                ], f)
