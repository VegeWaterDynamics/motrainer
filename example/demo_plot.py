import pickle
import numpy as np
from ml_lsmodel_ascat.plot import plot_gsdata, plot_tsdata

if __name__ == "__main__":
    # Manual input
    out_path = './results'
    file_data_gs = './example_data/testdata_plot_mean_sig_slop_curv'
    file_data_ts = './example_data/testdata_plot_timeseries_toy'
    # Visualization time-series plot
    with open(file_data_ts, 'rb') as f:
        df_plot_ts = pickle.load(f)
    df_plot_ts = df_plot_ts[1:800]
    ls_plot_ts = [df_plot_ts]*6  # repeat 6 times
    linecolor = {'TS1': 'b', 'TS3': 'k'}
    figsize = (42,12)

    plot_tsdata(ls_plot_ts,
                nrowcol=(2, 3),
                outpath='./testplot_ts.jpg',
                figsize=figsize,
                linecolor=linecolor,
                fontsize=32,
                rowlabels=['Row1', 'Row2'])


    # Visualization geo-spatial plot
    with open(file_data_gs, 'rb') as f:
        df_plot_gs = pickle.load(f)

    cbartext = 'Test colorbar notation \n test test test'
    title_lists = [
        r'$\sigma_{40}^o$ [dB]', r"$\sigma^{\prime}$ [dB/deg]",
        r'$\sigma^{\prime\prime}{\rm[dB/deg^2]}$'
    ]
    title_lists.extend(title_lists)
    rowlabel_list = ['mean', 'mean copy']
    kw_padding = {'pad': 5, 'h_pad': 5, 'w_pad': 3}

    plot_gsdata(df_plot_gs,
                nrowcol=(2, 3),
                outpath='./testplot_gs.jpg',
                titles=title_lists,
                rowlabels=rowlabel_list,
                cbar_mode='plot',
                kw_padding=kw_padding)

    
