# -*- coding: utf-8 -*-
import logging
import os
import time
import pickle
import numpy as np
import pandas as pd
import shap
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import regularizers
from keras.optimizers import Adam, Nadam, Adagrad, sgd
from keras.activations import relu, elu, tanh
from keras.losses import mean_squared_error, mean_squared_logarithmic_error,mean_absolute_error
from keras import backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
from skopt import gbrt_minimize, gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

logger = logging.getLogger(__name__)

global algoMethod
global color
global var, unit, input_variables

color = ['r', 'g', 'b', 'y']

algoMethod = 'DNN'
unit = ['[dB]','[dB/deg]','[dB/deg²]']

var = ['Backscatter', 'Slope', 'Curvature']
input_variables = ['LAI', 'GPP', 'WG2', 'RE']
output_variables = ['sig','slope','curv']

start_time = time.time()

class TrainDNN(object):
    def __init__(self, var_path):
        var_path = os.path.join('C:/Users/manue/OneDrive/Dokumente/Master Thesis - Susan/Publication/Data Output per Cluster/',var_name)
        with open(var_path, 'rb') as vp:
            df_all_gpi = pickle.load(vp)
        self.df_all_gpi = df_all_gpi[0]
        
    def pre_normalization(self, data, pre_Method):
        output = []
        #min_max = MinMaxScaler()
        #pre_Method = StandardScaler()
        x = pre_Method.fit_transform((data.values).reshape(-1,1))
        #x = stand.fit_transform((data.values).reshape(-1,1))
        for i in range(len(x)):
            y = x[i]
            output.append(y)

        return output
        

    def pretreatment(cluster):
        index = cluster.index
        for i in range(len(cluster.index)):
            temp = cluster[index[i]]
            temp = temp.dropna(axis=0,how='any')
            cluster[cluster.index[i]] = temp
        j = 0
        while True:
            if len(cluster[index[j]].LAI) == 0:
                cluster = cluster.drop(index[j])
            j += 1
            if j+1 == len(cluster.index):
                break
        cluster_pretreatment = cluster.copy()
        return cluster_pretreatment
            
    #-----------------------------------------------------------------------------#
    # Extract the labels and input data for the dnn
    # for sig
    def get_data_labels(data):
        data_norm = pd.DataFrame()
        data_norm['LAI'] = data['LAI'].values
        data_norm['WG2'] = data['WG2'].values
        data_norm['GPP'] = data['GPP'].values
        data_norm['RE'] = data['RE'].values
        
        #data_norm['slope']= data['slope'].values
    #    data_norm['curv']= data['curv'].values

        label_norm = pd.DataFrame()
        label_norm['sig']= data['sig'].values
        #for a model which can produce 3 paras simutaneously
        label_norm['slope']= data['slope'].values
        label_norm['curv']= data['curv'].values
        
        lables = label_norm.values
        input_set = data_norm.values
        
        return lables, input_set

    #-----------------------------------------------------------------------------#


    def keras_dnn(learning_rate, num_dense_layers,num_input_nodes,
                    num_dense_nodes, activation):
        #start the model making process and create our first layer
        model = Sequential()
        model.add(Dense(num_input_nodes, 
                        input_shape=(4,), 
                        activation=activation
                    ))
        #Notice that there the input_shape is important!!!!!!
        #create a loop making a new dense layer for the amount passed to this model.
        #naming the layers helps avoid tensorflow error deep in the stack trace.
        #no pooling/conv because not a figure processing!
        for i in range(num_dense_layers):
            name = 'layer_dense_{0}'.format(i+1)
            model.add(Dense(num_dense_nodes,
                    activation=activation,
                            name=name
                    ))
        #add our classification layer.
        model.add(Dense(units = 3))
        
        #setup our optimizer and compile
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss= mean_squared_error,
                    metrics=['mae', 'acc'])
        return model
        
    #-----------------------------------------------------------------------------#


    #-----------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------#

    # Start the training loop for each GPI 
    #save_at = r'C:/Users/manue/OneDrive/Dokumente/Master Thesis - Susan/Publication/Data Output per Cluster/'+algoMethod+'_'

    best_accuracy = 0.7
    best_loss = 1

    def DNN_Opt(save_at, agri_all, GPI):
        
        dim_learning_rate = Real(low=5e-4, high=1e-2, prior='log-uniform',
                                name='learning_rate')
        dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
        dim_num_input_nodes = Integer(low=1, high=128, name='num_input_nodes')
        dim_num_dense_nodes = Integer(low=1, high=128, name='num_dense_nodes')
        dim_activation = Categorical(categories=['relu', 'tanh'],
                                    name='activation')
        dim_batch_size = Integer(low=7, high=365, name='batch_size')
        #dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
        
        dimensions = [dim_learning_rate,
                    dim_num_dense_layers,
                    dim_num_input_nodes,
                    dim_num_dense_nodes,
                    dim_activation,
                    dim_batch_size
                    ]
        default_parameters = [1e-3, 1,128, 13, 'relu',64]
        
        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, num_dense_layers, num_input_nodes, 
                    num_dense_nodes,activation, batch_size):
            print('lr: {0:.1e}'.format(learning_rate)+\
                ' | num layers: '+str(num_dense_layers)+\
                ' | num nodes: '+str(num_dense_nodes)+\
                ' | activation: '+str(activation)+\
                ' | batch size: '+str(batch_size))
        
        
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
            model = keras_dnn(learning_rate=learning_rate,
                                num_dense_layers=num_dense_layers,
                                num_input_nodes=num_input_nodes,
                                num_dense_nodes=num_dense_nodes,
                                activation=activation
        
                                )
            
            #named blackbox becuase it represents the structure
            blackbox = model.fit(x=X_train,
                                y=Y_train,
                                epochs=300,
                                batch_size=batch_size,
                                validation_split=0.10,
                                callbacks=[es],
                                verbose=0
                                )
            

            accuracy = blackbox.history['val_acc'][-1]
            loss = blackbox.history['val_loss'][-1]
        
            nonlocal best_loss
            
            if loss < best_loss:
                # Save the new model to harddisk.
                #model.save('{}optimized_model_{}'.format(save_at,GPI))#new_folder
                model.save('{}optimized_model_{}_{}'.format(new_folder,test_index[0]+2010,GPI))
                best_loss = loss
        
            # Delete the Keras model with these hyper-parameters from memory.
            del model
            
            # Clear the Keras session, otherwise it will keep adding new
            # models to the same TensorFlow graph each time we create
            # a model with a different set of hyper-parameters.
            K.clear_session()
            #tensorflow.reset_default_graph()
            
            # the optimizer aims for the lowest score, so we return our negative accuracy
            return loss
        
        agri_all.index = pd.to_datetime(agri_all.index.values)
        
        df_all_gpi_norm = pd.DataFrame()
        

        pre_Method_LAI = StandardScaler()
        pre_Method_WG2 = StandardScaler()
        pre_Method_GPP = StandardScaler()
        pre_Method_RE = StandardScaler()
        
        pre_Method_sig = StandardScaler()
        pre_Method_slope = StandardScaler()
        pre_Method_curv = StandardScaler()
        
        # normalization did by PCA!!!
        df_all_gpi_norm['LAI'] = pre_normalization(agri_all['LAI'], pre_Method_LAI)
        df_all_gpi_norm['WG2'] = pre_normalization(agri_all['WG2'], pre_Method_WG2)
        df_all_gpi_norm['GPP'] = pre_normalization(agri_all['GPP'], pre_Method_GPP)
        df_all_gpi_norm['RE'] = pre_normalization(agri_all['RE'], pre_Method_RE)

        
        df_all_gpi_norm['sig'] = pre_normalization(agri_all['sig'], pre_Method_sig)
        df_all_gpi_norm['slope'] = pre_normalization(agri_all['slop'], pre_Method_slope)
        df_all_gpi_norm['curv'] = pre_normalization(agri_all['curv'], pre_Method_curv)
    
        df_all_gpi_norm['lon'] = agri_all['lon'].values
        df_all_gpi_norm['lat'] = agri_all['lat'].values
        
        df_all_gpi_norm.index = agri_all.index
        
        val_split_year = 2017
        jackknife_all = df_all_gpi_norm[df_all_gpi_norm.index.year < val_split_year]
        vali_all = df_all_gpi_norm[df_all_gpi_norm.index.year >= val_split_year]

        jackknife_metrics = np.zeros([3,4,7]) 
        loo = LeaveOneOut()
        

        
        new_folder = save_at+GPI+'/'
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        
        for train_index, test_index in loo.split(jackknife_all.copy().resample('Y').mean().index.year):
    
            
            train_all = jackknife_all[(jackknife_all.index.year != test_index[0]+2010)]
            test_all  = jackknife_all[(jackknife_all.index.year == test_index[0]+2010)]
            
            lables_train_val, data_train_val = get_data_labels(train_all)
            lables_test, data_test = get_data_labels(test_all)
            
            X_train =data_train_val
            Y_train = lables_train_val
            
            best_loss = 5
            
            gp_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    n_calls=15,
                                    noise= 0.01,
                                    n_jobs=-1,
                                    kappa = 5,
                                    x0=default_parameters)

            with open('{}Hyperparameter_space_{}_{}'.format(new_folder,test_index[0]+2010,GPI), 'wb') as f:
            #pdb.set_trace()
                pickle.dump([sorted(zip(gp_result.func_vals, gp_result.x_iters))], f)
            
            #'{}optimized_model_{}_{}'.format(new_folder,test_index[0]+2010,GPI)
            model =  load_model('{}optimized_model_{}_{}'.format(new_folder,test_index[0]+2010,GPI))
            #accuracy_score = model.evaluate(data_test, lables_test, steps=1)["accuracy"]
            predicted = model.predict(data_test)
            
            #rmse, mae, pearsCor, spearCor = np.zeros([3,]), np.zeros([3,]),np.zeros([3,]), np.zeros([3,])
            
            for j in range(3):
                if j == 0: 
                    pre_Method = pre_Method_sig
                if j == 1:
                    pre_Method = pre_Method_slope
                if j == 2:
                    pre_Method = pre_Method_curv
                re_predicted = np.asarray(pre_Method.inverse_transform(predicted[:,j].reshape(-1,1)), 'f')
                re_label = np.asarray(pre_Method.inverse_transform(lables_test[:,j].reshape(-1,1)), 'f')
                
                # rmse, mae, pearson, spearson
                jackknife_metrics[j, 0, test_index] = np.round(np.sqrt(((np.concatenate(re_predicted) - np.concatenate(re_label)) ** 2).mean()),5)
                jackknife_metrics[j, 1, test_index] = np.round(np.sqrt((np.abs(np.concatenate(re_predicted) - np.concatenate(re_label))).mean()),5)
                pearsCor_t = pearsonr(np.asarray(re_predicted,dtype='float').reshape([-1,]), np.asarray(re_label,dtype='float').reshape([-1,]))
                spearCor_t = spearmanr(np.asarray(re_predicted,dtype='float').reshape([-1,]), np.asarray(re_label,dtype='float').reshape([-1,]))
                
                jackknife_metrics[j, 2, test_index] = pearsCor_t[0]
                jackknife_metrics[j, 3, test_index] = spearCor_t[0]

            #jackknife_performance.append(jackknife_metrics)
            #print(jackknife_metrics[:,:,test_index])
        with open('{}performance_{}_{}'.format(new_folder,test_index[0]+2010,GPI), 'wb') as f:
            #pdb.set_trace()
                pickle.dump(jackknife_metrics, f)
        
        #choose the best optimized model by using rmse of all variables!!!
        # argmin(rmse_sig+rmse_slop+rmse_curv)
        rmse_array = np.nansum(jackknife_metrics, axis = 0)[0,:] # index 0 is rmse!
        
        #print("min rmse = "+str(np.nanmin(rmse_array)))
        rmse_min_year = np.nanargmin(rmse_array)+2010
        best_model = load_model('{}optimized_model_{}_{}'.format(new_folder,rmse_min_year,GPI))
        df = open('{}Hyperparameter_space_{}_{}'.format(new_folder,rmse_min_year,GPI), 'rb')
        best_hyper = pickle.load(df)
        df.close()
        best_hyper = best_hyper[0]
        #best_hyper = load_model('{}Hyperparameter_space_{}_{}'.format(new_folder,rmse_min_year,GPI))
        
        best_model.save('{}best_opt_model_({})_{}'.format(new_folder,rmse_min_year,GPI))
        with open('{}Best_Hyperpara_space_{}_{}'.format(new_folder,rmse_min_year,GPI), 'wb') as f:
            #pdb.set_trace()
                pickle.dump(best_hyper, f)

        date = vali_all.index   
        lables_val, input_val = get_data_labels(vali_all)
        predicted = best_model.predict(input_val)
        
        lables_plot_whole, input_plot_whole = get_data_labels(df_all_gpi_norm)
        
        predicted_whole = best_model.predict(input_plot_whole)
        # model.evaluate
        
        # shap for the explanation of the best model ??????
        background = input_plot_whole[np.random.choice(input_plot_whole.shape[0], 1000, replace=False)]

        shap.initjs()
        e =shap.DeepExplainer(best_model,background)
        shap_values = e.shap_values(input_plot_whole)
        
        #pdb.set_trace()
        with open('{}Shap_values(2010-2018)_{}'.format(new_folder,GPI), 'wb') as f:
            #pdb.set_trace()
            pickle.dump([zip(shap_values[0][:,0], shap_values[0][:,1], shap_values[0][:,2], shap_values[0][:,3], 
                            shap_values[1][:,0],shap_values[1][:,1],shap_values[1][:,2], shap_values[1][:,3], 
                            shap_values[2][:,0],shap_values[2][:,1],shap_values[2][:,2], shap_values[2][:,3],)], f)
            
        K.clear_session()
        return gp_result

    


