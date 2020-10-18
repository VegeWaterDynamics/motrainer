import numpy as np 
import shap

def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                                1000,
                                                replace=False)]

    shap.initjs()
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)

    return shap_values