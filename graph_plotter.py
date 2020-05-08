import numpy as np
import matplotlib.pyplot as plt

def scatter(zero_mean, one_mean, save_loc):
    #dataset_processing_equal.partition_dataset()
    x_values = list(range(1, zero_mean.shape[0]+1))
    # x_values = list(range(0, iteration))
    title = 'Non-Depressed Vs Depressed Mean Log Mel Features'
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x_values, zero_mean, c='r', s=.5, label='Non_Depressed')
    axs.scatter(x_values, one_mean, c='b', s=.5, label='Depressed')

    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, 'LogMel Feature Values', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs.legend()
    plt.grid(b=None)
    axs[0].yaxis.set_ticks(np.arange(0, 1.1, 0.05))
    axs[0].xaxis.set_ticks(np.arange(0, mel_bins+1, 5))
    plt.show()
    fig.savefig(save_loc)
    plt.close('all')