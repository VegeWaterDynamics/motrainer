import shap
import logging
import keras
import sklearn
import skopt
import numpy as np
import pickle
from pathlib import Path
from skopt.space import Real, Categorical, Integer 
from scipy.stats.stats import pearsonr, spearmanr
from model import keras_dnn
from keras.models import load_model

logger = logging.getLogger(__name__)

class DNNTrain(object):
    def __init__(self, train_input, train_output, 
                 test_input, test_output,
#                 vali_input, vali_output, 
                 resultpath='.'):
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output
#        self.vali_input = vali_input
#        self.vali_output = vali_output
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
        # prenormalization for output (or label)
        self.scaler_test_output = sklearn.preprocessing.StandardScaler()
        self.test_output = self.scaler_test_output.fit_transform(self.test_output)
        
        self.scaler_test_input = sklearn.preprocessing.StandardScaler()
        self.test_input = self.scaler_test_input.fit_transform(self.test_input)
        
        self.scaler_train_output = sklearn.preprocessing.StandardScaler()
        self.train_output = self.scaler_train_output.fit_transform(self.train_output)
        
        self.scaler_train_input = sklearn.preprocessing.StandardScaler()
        self.train_input = self.scaler_train_input.fit_transform(self.train_input)
#        scaler = sklearn.preprocessing.StandardScaler()
#        self.train_input = scaler.fit_transform(self.train_input)
#        self.train_output = scaler.fit_transform(self.train_output)
#        self.test_input = scaler.fit_transform(self.test_input)
#        self.test_output = scaler.fit_transform(self.test_output)
#        self.vali_input = scaler.fit_transform(self.vali_input)
#        self.vali_output = scaler.fit_transform(self.vali_output)

    def optimize(self, 
                 best_loss=1, 
                 n_calls=15,
                 noise= 0.01,
                 n_jobs=-1,
                 kappa = 5,
                 validation_method = 0.2,
                 x0=[1e-3, 1 , 4, 13, 'relu', 64]):
        self.best_loss = best_loss

        @skopt.utils.use_named_args(dimensions=list(self.dimensions.values()))
        def lossfunc(**dimensions):
            # setup model
            earlystop = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=0, patience=30)
            
            model = keras_dnn(dimensions, 
                              self.train_input.shape[1], 
                              self.train_output.shape[1])
            # Fit model
            blackbox = model.fit(x=self.train_input,
                                y=self.train_output,
                                batch_size=dimensions['batch_size'],
                                callbacks=[earlystop],
                                verbose=0,
                                validation_split=validation_method
                                )
            # Get loss
            loss = blackbox.history['val_loss'][-1]
            if loss < self.best_loss:
                self.model = model
                self.best_loss = loss
                self.hehe = 1
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
    
    # test/predict using new data
    def predict(self, input_data):
#        self.test_output = self.model.predict(self.test)
        output_data = self.model.predict(input_data)
        return output_data
    
    # performance calculation
    def performance(self, scaler, method, predicted):
#        scaler = self.scaler_test_output
        re_predicted = scaler.inverse_transform(predicted,'f')
        re_label     = scaler.inverse_transform(self.test_output,'f')
        
        difference = re_predicted - re_label
        performance = np.zeros([predicted.shape[1],1])
        if method == 'rmse':
            for j in range(predicted.shape[1]):
                performance[j,0] = np.round(np.sqrt(((difference[j]) ** 2).mean()),5)
        elif method == 'mae':
            for j in range(predicted.shape[1]):
                performance[j,0] = np.round((difference[j].mean()),5)
        elif method == 'pearson':
            for j in range(predicted.shape[1]):
                performance[j,0] = np.round(pearsonr(re_predicted[j], re_label[j]),5)[0]
        elif method == 'spearman':
            for j in range(predicted.shape[1]):
                performance[j,0] = np.round(spearmanr(re_predicted[j], re_label[j]),5)[0]
        return performance
    
    #=====================================================#
    def export_model(self, suffix_gpi='', suffix_year=''):
        path_model = self.resultpath# / 'model'
        self.model.save('{}/{}/optimized_model_{}'.format(path_model.as_posix(), suffix_gpi, suffix_year))
        with open('{}/{}/Hyperparameter_space_{}'.format(path_model.as_posix(), suffix_gpi, suffix_year), 'wb') as f:
            pickle.dump([sorted(zip(self.gp_result.func_vals, self.gp_result.x_iters))], f)
    
    def get_model(self, suffix_gpi='', suffix_year=''):
        path_model = self.resultpath# / 'model'
        model = load_model('{}/{}/optimized_model_{}'.format(path_model.as_posix(), suffix_gpi, suffix_year))
        return model
    
    # shap values calculation
    def get_shap_values(self, model, input_whole):
        background = input_whole[np.random.choice(input_whole.shape[0], 1000, replace=False)]
        
        shap.initjs()
        e = shap.DeepExplainer(model, background)
        self.shap_values = e.shap_values(input_whole)
        
        #return shap_values
    
    def export_shap_values(self, suffix_gpi='',suffix_year=''):
        path_model = self.resultpath# / 'shap_values'
        with open('{}/{}/Shap_values_{}'.format(path_model.as_posix(), suffix_gpi, suffix_year), 'wb') as f:
            pickle.dump([zip(self.shap_values[0][:,0], self.shap_values[0][:,1], self.shap_values[0][:,2], self.shap_values[0][:,3], 
                             self.shap_values[1][:,0],self.shap_values[1][:,1],self.shap_values[1][:,2], self.shap_values[1][:,3], 
                             self.shap_values[2][:,0],self.shap_values[2][:,1],self.shap_values[2][:,2], self.shap_values[2][:,3],)], f)

#=============================================================================#
# independent function
def save_performance(result_path, performance, method,
                         suffix_gpi):
    path_model = Path(result_path)# / 'shap_values'
    with open('{}/{}/{}'.format(path_model.as_posix(), 
              suffix_gpi, method), 'wb') as f:
        pickle.dump(performance, f)

def get_shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0], 1000, replace=False)]
        
    shap.initjs()
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)
        
    return shap_values
    
def export_shap_values(result_path, shap_values, suffix_gpi,suffix_year):
    path_model = Path(result_path)# / 'shap_values'
    with open('{}/{}/Shap_values_{}'.format(path_model.as_posix(), suffix_gpi, suffix_year), 'wb') as f:
        pickle.dump([zip(shap_values[0][:,0], shap_values[0][:,1], shap_values[0][:,2], shap_values[0][:,3], 
                         shap_values[1][:,0], shap_values[1][:,1], shap_values[1][:,2], shap_values[1][:,3], 
                         shap_values[2][:,0], shap_values[2][:,1], shap_values[2][:,2], shap_values[2][:,3],)], f)
