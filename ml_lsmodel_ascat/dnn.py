import logging
import tensorflow.keras
import sklearn
import skopt
import numpy as np
import pickle
from pathlib import Path
from skopt.space import Real, Categorical, Integer
from scipy.stats.stats import pearsonr, spearmanr
from tensorflow.keras.models import load_model
from ml_lsmodel_ascat.model import keras_dnn

logger = logging.getLogger(__name__)


class DNNTrain(object):
    def __init__(self, train_input, train_output, test_input, test_output):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
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

    def update_space(self, **kwrags):
        for key, value in kwrags.items():
            if key in ['learning_rate']:
                self.dimensions[key] = Real(low=value[0],
                                            high=value[1],
                                            prior='log-uniform',
                                            name=key)
            elif key in [
                    'num_dense_layers', 'num_input_nodes', 'num_dense_nodes',
                    'batch_size'
            ]:
                self.dimensions[key] = Integer(low=value[0],
                                               high=value[1],
                                               name=key)
            elif key in ['activation']:
                self.dimensions[key] = Categorical(categories=value, name=key)

    def normalize(self):
        # prenormalization for output (or label)
        self.scaler_test_output = sklearn.preprocessing.StandardScaler()
        self.test_output = self.scaler_test_output.fit_transform(
            self.test_output)

        self.scaler_test_input = sklearn.preprocessing.StandardScaler()
        self.test_input = self.scaler_test_input.fit_transform(self.test_input)

        self.scaler_train_output = sklearn.preprocessing.StandardScaler()
        self.train_output = self.scaler_train_output.fit_transform(
            self.train_output)

        self.scaler_train_input = sklearn.preprocessing.StandardScaler()
        self.train_input = self.scaler_train_input.fit_transform(
            self.train_input)

    def optimize(self,
                 best_loss=1,
                 n_calls=15,
                 noise=0.01,
                 n_jobs=-1,
                 kappa=5,
                 validation_method=0.2,
                 x0=[1e-3, 1, 4, 13, 'relu', 64]):
        self.best_loss = best_loss

        @skopt.utils.use_named_args(dimensions=list(self.dimensions.values()))
        def lossfunc(**dimensions):
            # setup model
            earlystop = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                      mode='min',
                                                      verbose=0,
                                                      patience=30)

            model = keras_dnn(dimensions, self.train_input.shape[1],
                              self.train_output.shape[1])
            # Fit model
            blackbox = model.fit(x=self.train_input,
                                 y=self.train_output,
                                 batch_size=dimensions['batch_size'],
                                 callbacks=[earlystop],
                                 verbose=0,
                                 validation_split=validation_method)
            # Get loss
            loss = blackbox.history['val_loss'][-1]
            if loss < self.best_loss:
                self.model = model
                self.best_loss = loss
                self.hehe = 1
            del model
            tensorflow.keras.backend.clear_session()
            return loss

        self.gp_result = skopt.gp_minimize(func=lossfunc,
                                           dimensions=list(
                                               self.dimensions.values()),
                                           n_calls=n_calls,
                                           noise=noise,
                                           n_jobs=n_jobs,
                                           kappa=kappa,
                                           x0=x0)

    # performance calculation
    def get_performance(self, scaler, method):
        # Temporally SL the model because of the TF graph execution issue
        self.model.save('/tmp/tmp_model')
        self.model = load_model('/tmp/tmp_model')
        predicted = self.model.predict(self.test_input)
        re_predicted = scaler.inverse_transform(predicted, 'f')
        re_label = scaler.inverse_transform(self.test_output, 'f')

        difference = re_predicted - re_label
        performance = np.zeros([predicted.shape[1], 1])
        if method == 'rmse':
            for j in range(predicted.shape[1]):
                performance[j,
                            0] = np.round(np.sqrt(((difference[j])**2).mean()),
                                          5)
        elif method == 'mae':
            for j in range(predicted.shape[1]):
                performance[j, 0] = np.round((difference[j].mean()), 5)
        elif method == 'pearson':
            for j in range(predicted.shape[1]):
                performance[j, 0] = np.round(
                    pearsonr(re_predicted[j], re_label[j]), 5)[0]
        elif method == 'spearman':
            for j in range(predicted.shape[1]):
                performance[j, 0] = np.round(
                    spearmanr(re_predicted[j], re_label[j]), 5)[0]
        self.performance = performance

    def export(self,
               path_model=None,
               path_hyperparameters=None,
               path_performance=None):

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

        if path_performance is not None:
            Path(path_performance).parent.mkdir(parents=True, exist_ok=True)
            with open(path_performance, 'wb') as f:
                pickle.dump(self.performance, f)
