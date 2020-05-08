import pickle
import os
import matplotlib.pyplot as plt


def evaluation_plot(path_to_datafold, save_directory, show_plot=False):
    """
    Plots the training verus validation True Negative, False Positive,
    False Negative, and True Positives from an experiment

    Inputs
        path_to_datafold: str - The location of the results for a specific fold
        save_directory: str - Location to save the plot
        show_plot: bool - Set True to visualise the plot, otherwise just save it
    """
    with open(path_to_datafold, 'rb') as f:
        complete_results = pickle.load(f)

    specific_res = complete_results[['train_true_negative',
                                     'train_false_positive',
                                     'train_false_negative',
                                     'train_true_positive',
                                     'val_true_negative',
                                     'val_false_positive',
                                     'val_false_negative',
                                     'val_true_positive']]
    specific_res_list = []
    for i in range(len(specific_res.columns)):
        specific_res_list.append(specific_res.iloc[:, i].values.tolist())

    fig, ax1 = plt.subplots(1, 2, sharex=True)
    x_values = list(range(1, 1+len(specific_res_list[0])))

    ax1[0].plot(x_values, specific_res_list[0], label='train_true_negative')
    ax1[0].plot(x_values, specific_res_list[1], label='train_false_positive')
    ax1[0].plot(x_values, specific_res_list[2], label='train_false_negative')
    ax1[0].plot(x_values, specific_res_list[3], label='train_true_positive')
    ax1[1].plot(x_values, specific_res_list[4], label='val_true_negative')
    ax1[1].plot(x_values, specific_res_list[5], label='val_false_positive')
    ax1[1].plot(x_values, specific_res_list[6], label='val_false_negative')
    ax1[1].plot(x_values, specific_res_list[7], label='val_true_positive')

    ax1[0].legend(shadow=False)
    ax1[1].legend(shadow=False)

    ax1[0].set_xlabel('Epoch')
    ax1[1].set_xlabel('Epoch')
    ax1[0].set_ylabel('Count')
    ax1[1].set_ylabel('Count')
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    if show_plot:
        plt.show()

    save_loc = os.path.join(save_directory, 'classification_plot.png')
    fig.savefig(save_loc)

