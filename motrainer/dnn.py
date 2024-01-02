import logging

# disable WARNING:absl:Found untraced functions such as _update_step_xla while saving
# see https://github.com/tensorflow/tensorflow/issues/47554
import absl.logging
import h5py
import skopt
import tensorflow as tf
from skopt.space import Categorical, Integer, Real

from motrainer.model import keras_dnn, keras_dnn_lossweight

absl.logging.set_verbosity(absl.logging.ERROR)


logger = logging.getLogger(__name__)


class NNTrain:
    """Neuron Network trainning object.

    Methods
    -------
    update_space(**kwrags)
        Update searching space of tranning
    optimize(best_loss=1, n_calls=15, epochs=100, noise=0.01, n_jobs=-1,
             kappa=5, validation_split=0.2, x0=[1e-3, 1, 4, 13, 'relu', 64],
             training_method='dnn', loss_weights=None, verbose=0):
        Optimize the neuron network within the searching space by given
        optimization settings
    """

    def __init__(self, train_input, train_output):
        """Initialize NNTrain object.

        Parameters
        ----------
        train_input : pandas.DataFrame
            Input data of trainning
        train_output : pandas.DataFrame
            Output data of trainning
        """
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
        """Update searching space of optimization."""
        for key, value in kwrags.items():
            logger.debug(f'Update seaching sapce: {key}={value}')
            # skopt.space instances
            if isinstance(value, Real | Categorical | Integer):
                self.dimensions[key] = value
                self.dimensions[key].name = key

            # Float searching space, e.g. learning rates
            elif any([isinstance(obj, float) for obj in value]):
                assert len(value) == 2
                if any([isinstance(obj, int) for obj in value]):
                    logger.warning(
                        f'Mixed fload/int type found in {key}:{value}. '
                        'The search space will be interpreted as float. '
                        'If this behavior is not desired, try to specify'
                        f'all elements in {key} with the same type.')
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
                    f'Do not understand searching space: {key}:{value}.')
                raise NotImplementedError

    def optimize(self,
                 best_loss=1.,
                 n_calls=15,
                 epochs=100,
                 noise=0.01,
                 n_jobs=-1,
                 kappa=5,
                 validation_split=0.2,
                 x0=None,
                 training_method='dnn',
                 loss_weights=None,
                 verbose=0):
        """Optimize the neuron network within the searching space..

        Parameters
        ----------
        best_loss : float, optional
            Threshold of loss, by default 1.
            If the final loss is larger than best_loss, the model won't be
            accepted.
        n_calls : int, optional
            Total number of evaluations, by default 15
        epochs : int, optional
            Number of epochs to train the model, by default 100
        noise : float, optional
            Variance of input, by default 0.01
        n_jobs : int, optional
            Number of cores to run in parallel, by default -1, which sets the
            number of jobs equal to the number of cores.
        kappa : int, optional
            Controls how much of the variance in the predicted values should be
             taken into account. , by default 5
        validation_split : float, optional
            Float between 0 and 1. Fraction of the training data to be used as
            validation data. , by default 0.2
        x0 : list, optional
            Initial input points., by default [1e-3, 1, 4, 13, 'relu', 64]
        training_method : str, optional
            Traning method selection. Select from 'dnn' or 'dnn_lossweights'.
            By default 'dnn'
        loss_weights : list, optional
            Scalar coefficients (Python floats) to weight the loss
            contributions of different model outputs
            By default None, which means equal weights of all outputs
        verbose : int, optional
            Control the verbosity.
            By default 0, which means no screen feedback.
        """
        self.best_loss = best_loss
        self.keras_verbose = verbose
        self.loss_weights = loss_weights

        @skopt.utils.use_named_args(dimensions=list(self.dimensions.values()))
        def func(**dimensions):
            logger.info(f'optimizing with dimensions: {dimensions}')

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
                                   f'Using default weights {self.loss_weights}')
                model = keras_dnn_lossweight(dimensions,
                                             self.train_input.shape[1],
                                             self.train_output.shape[1],
                                             self.loss_weights)
                train_output = [
                    self.train_output.iloc[:, i]
                    for i in range(self.train_output.shape[1])
                ]

            # Fit model
            history = model.fit(x=self.train_input,
                                y=train_output,
                                epochs=epochs,
                                batch_size=dimensions['batch_size'],
                                callbacks=[earlystop],
                                verbose=self.keras_verbose,
                                validation_split=validation_split)

            # Get loss
            loss = history.history['val_loss'][-1]
            if loss < self.best_loss:
                self.model = model
                self.best_loss = loss
                self.history = history.history
            del model
            tf.keras.backend.clear_session()
            return loss

        if x0 is None:
            x0 = [1e-3, 1, 4, 13, 'relu', 64]

        self.gp_result = skopt.gp_minimize(func=func,
                                           dimensions=list(
                                               self.dimensions.values()),
                                           n_calls=n_calls,
                                           noise=noise,
                                           n_jobs=n_jobs,
                                           kappa=kappa,
                                           x0=x0)

    def export(self, path_model=None, meta_data=None):
        """Export model, hyperparameters and training metadata."""
        self.model.save(f"{path_model}.h5", save_format='h5')

        hyperparameters = sorted(
            zip(
                self.gp_result.func_vals,
                self.gp_result.x_iters,
                strict=True
                )
        )

        # open the hdf file and write the hyperparameters as attributes
        with h5py.File(f"{path_model}.h5", 'a') as f:
            # list of tuples are not supported by hdf5, so convert to string
            f.attrs['hyperparameters'] = str(hyperparameters)

            # add meta data to the hdf file
            if meta_data is not None:
                for key, value in meta_data.items():
                    f.attrs[key] = value
