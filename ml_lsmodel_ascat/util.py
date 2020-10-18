import numpy as np
import shap
from scipy.stats.stats import pearsonr, spearmanr


def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                              1000,
                                              replace=False)]

    shap.initjs()
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)

    return shap_values


def performance(data_input, data_output, model, scaler, method):
    predicted = model.predict(data_input)
    re_predicted = scaler.inverse_transform(predicted, 'f')
    re_label = scaler.inverse_transform(data_output, 'f')

    difference = re_predicted - re_label
    perf = np.zeros([predicted.shape[1], 1])
    if method == 'rmse':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(np.sqrt(((difference[j])**2).mean()), 5)
    elif method == 'mae':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round((difference[j].mean()), 5)
    elif method == 'pearson':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(pearsonr(re_predicted[j], re_label[j]), 5)[0]
    elif method == 'spearman':
        for j in range(predicted.shape[1]):
            perf[j, 0] = np.round(spearmanr(re_predicted[j], re_label[j]),
                                  5)[0]

    return perf
