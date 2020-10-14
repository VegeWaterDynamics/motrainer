import logging
import keras
import sklearn
import skopt
import numpy as np
import pickle
from pathlib import Path
from skopt.space import Real, Categorical, Integer 


logger = logging.getLogger(__name__)


class DNNTrain(object):
    def __init__(self, train_input, train_output, 
                 test_input, test_output,
                 vali_input, vali_output, resultpath='.'):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
        self.vali_input = vali_input
        self.vali_output = vali_output
        self.resultpath = Path(resultpath)
        self.resultpath.mkdir(parents=True, exist_ok=True)
        self.dimensions = {'learning_rate': Real(low=5e-4, high=1e-2, prior='log-uniform',name='learning_rate'), 
                      'num_dense_layers': Integer(low=1, high=2, name='num_dense_layers'),
                      'num_input_nodes': Integer(low=2, high=6, name='num_input_nodes'),
                      'num_dense_nodes': Integer(low=1, high=128, name='num_dense_nodes'),
                      'activation': Categorical(categories=['relu'], name='activation'),
                      'batch_size': Integer(low=7, high=365, name='batch_size')
                }

    def update_space(self, **kwrags):
        for key, value in kwrags.items():
            if key in ['learning_rate']:
                self.dimensions[key] = Real(low=value[0], high=value[1], prior='log-uniform', name=key)
            elif key in ['num_dense_layers', 'num_input_nodes', 'num_dense_nodes', 'batch_size']:
                self.dimensions[key] = Integer(low=value[0], high=value[1], name=key)
            elif key in ['activation']:
                self.dimensions[key] = Categorical(categories=value, name=key)

    def normalize(self):
        scaler = sklearn.preprocessing.StandardScaler()
        self.train_input = scaler.fit_transform(self.train_input)
        self.train_output = scaler.fit_transform(self.train_output)
        self.test_input = scaler.fit_transform(self.test_input)
        self.test_output = scaler.fit_transform(self.test_output)
        self.vali_input = scaler.fit_transform(self.vali_input)
        self.vali_output = scaler.fit_transform(self.vali_output)

    def optimize(self, 
                 best_loss=1, 
                 n_calls=15,
                 noise= 0.01,
                 n_jobs=-1,
                 kappa = 5,
                 x0=[1e-3, 1 , 4, 13, 'relu', 64]):
        self.best_loss = best_loss

        @skopt.utils.use_named_args(dimensions=list(self.dimensions.values()))
        def lossfunc(**dimensions):
            # setup model
            earlystop = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=0, patience=30)
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(dimensions['num_input_nodes'], 
                    input_shape=(self.train_input.shape[1],), 
                    activation=dimensions['activation']))

            for i in range(dimensions['num_dense_layers']):
                name = 'layer_dense_{0}'.format(i+1)
                model.add(keras.layers.Dense(dimensions['num_input_nodes'],
                        activation=dimensions['activation'],
                        name=name))
            model.add(keras.layers.Dense(units = self.train_output.shape[1]))
            adam = keras.optimizers.Adam(lr=dimensions['learning_rate'])
            model.compile(optimizer=adam, loss= keras.losses.mean_squared_error, metrics=['mae', 'acc'])
            # Fit model
            blackbox = model.fit(x=self.train_input,
                                y=self.train_output,
                                batch_size=dimensions['batch_size'],
                                callbacks=[earlystop],
                                verbose=0
                                )
            # Get loss
            loss = blackbox.history['loss'][-1]
            if loss < self.best_loss:
                self.model = model
                self.best_loss = loss
            del model
            keras.backend.clear_session()
            return loss
    
        self.gp_result = skopt.gp_minimize(func=lossfunc,
                                    dimensions=list(self.dimensions.values()),
                                    n_calls=n_calls,
                                    noise= noise,
                                    n_jobs=n_jobs,
                                    kappa = kappa,
                                    x0=x0)

    def export(self, suffix=''):
        path_model = self.resultpath / 'model'
        self.model.save('{}/model/optimized_model_{}'.format(path_model.as_posix(), suffix))
        with open('{}/model/Hyperparameter_space_{}'.format(path_model.as_posix(), suffix), 'wb') as f:
            pickle.dump([sorted(zip(self.gp_result.func_vals, gp_result.x_iters))], f)
    
    