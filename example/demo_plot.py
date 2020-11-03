import pickle
import numpy as np
from ml_lsmodel_ascat.plot import plot_gsdata

if __name__ == "__main__":
    # Manual input
    out_path = './results'
    file_data = './example_data/testdata_plot_mean_sig_slop_curv'


    with open(file_data, 'rb') as f:
        df_plot = pickle.load(f)

    # Visualization
    cbartext = 'Test colorbar notation \n test test test'
    title_lists = [
        r'$\sigma_{40}^o$ [dB]', r"$\sigma^{\prime}$ [dB/deg]",
        r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$'
    ]
    title_lists.extend(title_lists)
    rowlabel_list = ['mean', 'mean copy']
    kw_padding = {'pad': 5, 'h_pad': 5, 'w_pad': 3}

    plot_gsdata(df_plot,
                nrowcol=(2, 3),
                titles=title_lists,
                rowlabels=rowlabel_list,
                cbar_mode='plot',
                kw_padding=kw_padding)
