import importlib
import h5py
import numpy as np
import pickle
import os
import logging.handlers
import matplotlib.pyplot as plt
import math
import time
import audio_feature_extractor as afe
import utilities as util
import config_dataset


def stem_plot_combined(data, save_loc, title):
    fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True)
    comp_data = [j for i in data for j in i]
    x_axis = list(range(1, 129))
    x_axis = x_axis * 4
    axs.stem(x_axis, comp_data, use_line_collection=True)

    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, f"LogMel Feature Values - {type}", va='center', rotation='vertical')
    fig.text(0.5, 0.99, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.show()
    fig.savefig(save_loc)
    plt.close()


def stem_plot(data, save_loc, title):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True)
    for pointer, i in enumerate(data):
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        axs[a[pointer]][b[pointer]].stem(i, use_line_collection=True)
        axs[a[pointer]][b[pointer]].set_title(f"Fold_{pointer + 1}")

    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, f"LogMel Feature Values - {type}", va='center', rotation='vertical')
    fig.text(0.5, 0.99, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.show()
    fig.savefig(save_loc)
    plt.close()


def plot_individual_files(data, type, save_loc):
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    lengths = []
    for pointer, i in enumerate(data):
        lengths.append(len(i))
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        for j in data[pointer]:
            axs[a[pointer]][b[pointer]].scatter(j, list(range(1, 129)), s=.5)
        axs[a[pointer]][b[pointer]].set_title(f"{type}_Fold_{pointer + 1}_Individual_Files")

    fig.text(0.5, 0.004, f"LogMel Feature Values - {type}", ha='center')
    fig.text(0.004, 0.5, 'Mel Frequency Bins', va='center', rotation='vertical')
    fig.text(0.5, 0.985, f"Individual Files {type}", ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.show()
    fig.savefig(save_loc)
    plt.close()


def plot_figure(mean, var, std, xlabel, save_loc, title):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    # data_types = [mean, var, std]
    data_types = [mean, std]
    if isinstance(data_types[0], list):
        lengths = []
        for pointer, i in enumerate(data_types):
            lengths.append(len(i))
            a = [0, 1]
            for j in data_types[pointer]:
                # y by x (Rows by Columns)
                axs[a[pointer]].scatter(j, list(range(1, 129)), s=.5)
        fig.text(0.5, 0.004, f"LogMel Feature Values", ha='center')
        fig.text(0.004, 0.5, 'Mel Frequency Bins', va='center', rotation='vertical')
        fig.text(0.5, 0.975, title, ha='center')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        plt.show()
    #     ax1.scatter(list(range(1, lengths[0] + 1)), mean[0], 'r')
    #     ax1.scatter(list(range(1, lengths[1] + 1)), mean[1], 'g')
    #     ax1.scatter(list(range(1, lengths[2] + 1)), mean[2], 'b')
    #     ax1.scatter(list(range(1, lengths[3] + 1)), mean[3], 'y')
    #     ax2.scatter(list(range(1, lengths[0] + 1)), var[0], 'r')
    #     ax2.scatter(list(range(1, lengths[1] + 1)), var[1], 'g')
    #     ax2.scatter(list(range(1, lengths[2] + 1)), var[2], 'b')
    #     ax2.scatter(list(range(1, lengths[3] + 1)), var[3], 'y')
    #     ax3.scatter(list(range(1, lengths[0] + 1)), std[0], 'r')
    #     ax3.scatter(list(range(1, lengths[1] + 1)), std[1], 'g')
    #     ax3.scatter(list(range(1, lengths[2] + 1)), std[2], 'b')
    #     ax3.scatter(list(range(1, lengths[3] + 1)), std[3], 'y')
    # else:
    #     x = list(range(1, len(mean) + 1))
    #     ax1.scatter(x, mean, 'r')
    #     ax2.scatter(x, var, 'b')
    #     ax3.scatter(x, std, 'g')
    #
    # ax1.set_title('Mean')
    # ax2.set_title('Variance')
    # ax3.set_title('Standard Dev')
    # ax1.set(xlabel=xlabel)
    # ax2.set(xlabel=xlabel)
    # ax3.set(xlabel=xlabel)

    plt.show()
    fig.savefig(save_loc)
    plt.close()


def calc_mean(data):
    numerator = 0
    for k in data:
        numerator += k

    return numerator / len(data)


def calc_variance(data, mean):
    x = 0
    for k in data:
        x += (k - mean) ** 2

    return x / (len(data) - 1)


def calc_standard_deviation(data):
    # std = []
    # for i in data:
    #     std.append(math.sqrt(i))
    return np.sqrt(data)


def setup():
    stat_log_path = os.path.join(workspace, folder_name, 'stat_log')
    log_file = os.path.join(stat_log_path, 'stat_logger.log')
    if not os.path.exists(stat_log_path):
        os.mkdir(stat_log_path)
    elif os.path.exists(log_file):
        os.remove(log_file)
    my_logger = logging.getLogger('MyLogger')
    my_logger.setLevel(logging.INFO)
    my_handler = logging.handlers.RotatingFileHandler(log_file)
    my_logger.addHandler(my_handler)

    return my_logger, stat_log_path


def fold_stats():
    h5_file = h5py.File(data_dir, 'r')
    features = h5_file['features'][:]

    data_fold = os.path.join(workspace, folder_name, 'data_folds')
    folds = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
    fold_paths = []
    comp_indexes = []
    fold_mean_list = []
    fold_var_list = []
    fold_std_list = []
    individual_fold_mean = []
    individual_fold_var = []
    individual_fold_std = []
    for pointer, i in enumerate(folds):
        temp_path = os.path.join(data_fold, i+'.pickle')
        fold_paths.append(temp_path)
        my_logger.info(f"Fold Path: {temp_path}")
        with open(temp_path, 'rb') as f:
            data = pickle.load(f)
        indexes = data[-1].tolist()
        comp_indexes.append(indexes)

        temp_feats = features[indexes]
        temp_mean_list = []
        temp_var_list = []
        temp_std_list = []

        for j in temp_feats:
            reshaped = np.reshape(j[0], (mel_bins, j[0].shape[0]//mel_bins))
            temp_mean = np.mean(reshaped, axis=1)
            temp_var = np.var(reshaped, axis=1)
            temp_mean_list.append(temp_mean)
            temp_var_list.append(temp_var)
            temp_std_list.append(np.sqrt(temp_var))
        fold_mean_list.append(temp_mean_list)
        fold_var_list.append(temp_var_list)
        fold_std_list.append(temp_std_list)
        # my_logger.info(config_dataset.SEPARATOR)
        # my_logger.info(f"Fold_{pointer+1} mean:")
        # my_logger.info(fold_mean_list[pointer])
        # my_logger.info(f"Fold_{pointer+1} Variance")
        # my_logger.info(fold_var_list[pointer])
        # my_logger.info(f"Fold_{pointer+1} standard deviation")
        # my_logger.info(fold_std_list[pointer])
        # my_logger.info(config_dataset.SEPARATOR)

        # per_fold_mean = calc_mean(temp_mean_list)
        # per_fold_var = calc_variance(temp_mean_list, per_fold_mean)
        # per_fold_std = calc_standard_deviation(per_fold_var)
        # my_logger.info(f"Mean of Fold_{pointer+1}: {per_fold_mean}")
        # my_logger.info(f"Variance of Fold_{pointer+1}: {per_fold_var}")
        # my_logger.info(f"Standard Deviation of Fold_{pointer+1}: {per_fold_std}")
        # individual_fold_mean.append(per_fold_mean)
        # individual_fold_var.append(per_fold_var)
        # individual_fold_std.append(per_fold_std)
    type = 'mean'
    save_file = f"1_{type}_all_files"
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    plot_individual_files(fold_mean_list, type, save_loc)
    type = 'var'
    save_file = f"2_{type}_all_files"
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    plot_individual_files(fold_var_list, type, save_loc)
    type = 'std'
    save_file = f"3_{type}_all_files"
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    plot_individual_files(fold_std_list, type, save_loc)
    #plot_figure(fold_mean_list, fold_var_list, fold_std_list, xlabel, save_loc)

    my_logger.info(config_dataset.SEPARATOR)
    individual_fold_mean = []
    individual_fold_var = []
    individual_fold_std = []
    for i in range(data_folds):
        per_fold_mean = calc_mean(fold_mean_list[i])
        per_fold_var = calc_variance(fold_mean_list[i], per_fold_mean)
        per_fold_std = calc_standard_deviation(per_fold_var)
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"Mean of Fold_{i+1}: ")
        my_logger.info(f"{per_fold_mean}")
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"Variance of Fold_{i+1}: ")
        my_logger.info(f"{per_fold_var}")
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"Standard Deviation of Fold_{i+1}: ")
        my_logger.info(f"{per_fold_std}")
        individual_fold_mean.append(per_fold_mean)
        individual_fold_var.append(per_fold_var)
        individual_fold_std.append(per_fold_std)
    my_logger.info(config_dataset.SEPARATOR)

    xlabel = 'Fold'
    save_file = '4_mean_std_individual_folds'
    title = 'Individual Fold Analysis (All 4 Folds)'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    plot_figure(individual_fold_mean, individual_fold_var, individual_fold_std, xlabel, save_loc, title)

    comp_fold_mean = []
    comp_fold_var = []
    comp_fold_std = []
    for i in range(data_folds):
        folds_for_train = []
        for j in range(data_folds):
            if j == i:
                folds_for_dev = j
            else:
                folds_for_train.append(j)
        log_fold_train = [k+1 for k in folds_for_train]
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"Train Folds: {log_fold_train}")
        my_logger.info(f"Dev Fold: {folds_for_dev+1}")
        my_logger.info(config_dataset.SEPARATOR)

        temp_mean = [value for point in folds_for_train for value in
                     fold_mean_list[point]]

        fold_mean = calc_mean(temp_mean)
        fold_var = calc_variance(temp_mean, fold_mean)
        fold_std = calc_standard_deviation(fold_var)

        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"The mean of Fold_{i+1}: ")
        my_logger.info(f"{fold_mean}")
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"The variance of Fold_{i+1}: ")
        my_logger.info(f"{fold_var}")
        my_logger.info(config_dataset.SEPARATOR)
        my_logger.info(f"The Standard Deviation for Fold_{i+1}: ")
        my_logger.info(f"{fold_std}")
        comp_fold_mean.append(fold_mean)
        comp_fold_var.append(fold_var)
        comp_fold_std.append(fold_std)

    save_file = '5_mean_std_combination_folds (folds_234_134_124_123)'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    title = 'Combined Fold Analysis'
    plot_figure(comp_fold_mean, comp_fold_var, comp_fold_std, xlabel, save_loc, title)

    save_file = '6_stem_plot_mean_per_Individualfold'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot(individual_fold_mean, save_loc, save_file)
    save_file = '7_stem_plot_mean_per_Individualfold_combined'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot_combined(individual_fold_mean, save_loc, save_file)

    save_file = '8_stem_plot_std_per_Individualfold'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot(individual_fold_std, save_loc, save_file)
    save_file = '9_stem_plot_std_per_Individualfold_combined'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot_combined(individual_fold_std, save_loc, save_file)

    save_file = '10_stem_plot_mean_Datafolds_(234_134_124_123)'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot(comp_fold_mean, save_loc, save_file)
    save_file = '11_stem_plot_mean_Datafolds_(234_134_124_123)_combined'
    save_loc = os.path.join(stat_log_path, save_file + '.png')
    stem_plot_combined(comp_fold_mean, save_loc, save_file)

    save_file = '12_stem_plot_std_per_Datafold_(234_134_124_123)'
    save_loc = os.path.join(stat_log_path, save_file+'.png')
    stem_plot(comp_fold_std, save_loc, save_file)
    save_file = '13_stem_plot_std_Datafolds_(234_134_124_123)_combined'
    save_loc = os.path.join(stat_log_path, save_file + '.png')
    stem_plot_combined(comp_fold_std, save_loc, save_file)


def plot_class_graph_error_bars_combined(zero_mean, zero_std, one_mean,
                                         one_std, save_loc, title):

    x_values = list(range(1, zero_mean.shape[0]+1))
    fig, axs = plt.subplots(1, 1)
    axs.errorbar(x_values, zero_mean, yerr=zero_std, c='r',
                 label='Non-Depressed')
    axs.errorbar(x_values, one_mean, yerr=one_std, c='b',
                 label='Non-Depressed')
    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, 'LogMel Feature Values', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs.legend()
    length = zero_mean.shape[0]
    axs.xaxis.set_ticks(np.arange(0, length+1, 5))
    axs.grid(False)
    plt.show()
    fig.savefig(save_loc)
    plt.close('all')


def plot_class_graph_error_bars(mean, std, save_loc, title):
    x_values = list(range(1, mean.shape[0]+1))
    fig, axs = plt.subplots(1, 1)
    axs.errorbar(x_values, mean, yerr=std, c='r',
                 label='Non-Depressed')
    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, 'LogMel Feature Values', va='center', rotation='vertical')
    fig.text(0.5, 0.975, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs.legend()
    length = mean.shape[0]
    axs.xaxis.set_ticks(np.arange(0, length+1, 5))
    axs.grid(False)
    plt.show()
    fig.savefig(save_loc)
    plt.close('all')


def plot_class_graph(zero, one, save_loc=None, title=None):
    x_values = list(range(1, zero.shape[0]+1))
    fig, axs = plt.subplots(1, 1)
    axs.scatter(x_values, zero, c='r', s=.5, label='Non_Depressed')
    axs.scatter(x_values, one, c='b', s=.5, label='Depressed')
    # axs.errorbar(x_values, zero_mean, yerr=zero_std, c='r',
    #              label='Non-Depressed')

    fig.text(0.5, 0.004, 'Mel Frequency Bins', ha='center')
    fig.text(0.004, 0.5, 'LogMel Feature Values', va='center', rotation='vertical')
    if title:
        fig.text(0.5, 0.975, title, ha='center')
    fig.set_figheight(15)
    fig.set_figwidth(15)
    axs.legend()
    max_z = np.max(zero)
    max_o = np.max(one)
    max_value = np.max([max_z, max_o])
    min_z = np.min(zero)
    min_o = np.min(one)
    min_value = np.min([min_z, min_o])
    length = zero.shape[0]
    axs.yaxis.set_ticks(np.arange(min_value, max_value/1.2, 2.5))
    axs.xaxis.set_ticks(np.arange(0, length+1, 5))
    axs.grid(False)
    plt.show()
    if save_loc:
        fig.savefig(save_loc)
    plt.close('all')


def process_data(features, labels):
    segment_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    reshaped_feat = []
    # Work out how many dimensions the segmented feature dataset will have
    length = 0
    for pointer, i in enumerate(features[:, 0]):
        # Convert the 1D feature vector back to a 2D logmel feature vector
        dimension = i.shape[0] // mel_bins
        reshaped = np.reshape(i, (mel_bins, dimension))
        length += (dimension // segment_dim) + 1
        reshaped_feat.append(reshaped.tolist())

    update_features = np.zeros((length, mel_bins, segment_dim),
                               dtype=np.float32)

    pointer = feature_overlap_percent = 0
    final_folders = []
    final_classes = []
    final_indexes = []
    for i, data in enumerate(reshaped_feat):
        data = np.array(data)
        new_features, new_folders, new_classes, \
        new_indexes = afe.logmel_segmenter(data,
                                           labels[0][i],
                                           labels[1][i],
                                           segment_dim,
                                           feature_overlap_percent)
        z, y, x = new_features.shape
        update_features[pointer:pointer+z, :, :] = new_features
        new_indexes = new_indexes + pointer
        final_folders.append(new_folders)
        final_classes.append(new_classes)
        final_indexes.append(new_indexes.tolist())
        pointer += z

    final_folders = [j for i in final_folders for j in i]
    final_classes = [j for i in final_classes for j in i]
    final_indexes = [j for i in final_indexes for j in i]
    update_labels = [final_folders, final_classes, final_indexes]

    return update_features, update_labels


def euclidean_per_class_stats(mean, features):
    segment_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    reshaped_feat = []
    # Work out how many dimensions the segmented feature dataset will have
    length = 0
    for pointer, i in enumerate(features[:, 0]):
        # Convert the 1D feature vector back to a 2D logmel feature vector
        dimension = i.shape[0] // mel_bins
        reshaped = np.reshape(i, (mel_bins, dimension))
        length += (dimension // segment_dim) + 1
        reshaped_feat.append(reshaped.tolist())

    del reshaped
    update_features = np.zeros((length, mel_bins, segment_dim),
                               dtype=np.float32)
    pointer = 0
    for i, data in enumerate(reshaped_feat):
        data = np.array(data)
        new_features, _, _, _ = afe.logmel_segmenter(data, 0, 0, segment_dim)
        z, _, _ = new_features.shape
        update_features[pointer:pointer + z, :, :] = new_features
        pointer += z

    del data
    del new_features
    del reshaped_feat
    euclid_total = np.swapaxes(update_features, 1, 2)
    euclid_total = np.float64(euclid_total-mean)
    euclid_total = np.sum(np.sum(euclid_total**2, axis=1), axis=0)
    euclid_total = euclid_total ** 0.5
    euclid_total = euclid_total / (length * segment_dim)
    # min_val = np.min(euclid_total)
    # euclid_total = euclid_total / min_val
    # return euclid_total / (length * segment_dim)
    return euclid_total


def euclidean_stats(zero_mean, one_mean, features, ground_truth,
                    norm=np.array([])):
    """
    Function to determine how far each data point is from the correct class mean
    If class one and class two have a large distance between eachother the
    L2-Norm distance from a data point to each of the class means should
    result in the correct prediction of the data point. For example data
    point is from class one, therefore the L2-Norm should be a minimum when
    using the class one mean and a maximum when using the class zero mean
    :param zero_mean:
    :param one_mean:
    :param features:
    :param ground_truth:
    :return:
    """
    segment_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    reshaped_feat = []
    # Work out how many dimensions the segmented feature dataset will have
    length = 0
    for pointer, i in enumerate(features[:, 0]):
        # Convert the 1D feature vector back to a 2D logmel feature vector
        dimension = i.shape[0] // mel_bins
        reshaped = np.reshape(i, (mel_bins, dimension))
        if norm.size != 0:
            reshaped = reshaped / norm
        length += (dimension // segment_dim) + 1
        reshaped_feat.append(reshaped.tolist())

    del reshaped
    update_features = np.zeros((length, mel_bins, segment_dim),
                               dtype=np.float32)
    pointer = 0
    for i, data in enumerate(reshaped_feat):
        data = np.array(data)
        new_features, _, _, _ = afe.logmel_segmenter(data, 0, 0, segment_dim)
        z, _, _ = new_features.shape
        update_features[pointer:pointer + z, :, :] = new_features
        pointer += z

    del data
    del new_features
    del reshaped_feat
    euclid_total = np.zeros(mel_bins)
    correct = 0
    if ground_truth == 1:
        print('wait')
    for i in update_features:
        # difference = (i.transpose()-mean).transpose()
        difference_zero = np.sum((np.mean(i, axis=1) - zero_mean) ** 2)
        difference_one = np.sum((np.mean(i, axis=1) - one_mean) ** 2)

        minimum = np.array([difference_zero ** 0.5, difference_one ** 0.5])
        minimum = np.argmin(minimum)
        if minimum == ground_truth:
            correct += 1

    return correct / length


def euclidean_stats_2(zero_mean, one_mean, features, ground_truth):
    features = features ** 2
    summed = np.sum(features, axis=1)
    zero_mean = np.reshape(zero_mean, (-1, 1))
    one_mean = np.reshape(one_mean, (-1, 1))
    difference_zero = np.sqrt((features - zero_mean) ** 2)
    difference_one = np.sqrt((features - one_mean) ** 2)

    correct = np.zeros(mel_bins)
    for i in range(difference_zero.shape[1]):
        minimum = np.argmin((difference_zero[:, i], difference_one[:, i]),
                            axis=0)
        for j in range(mel_bins):
            if minimum[j] == ground_truth:
                correct[j] += 1

    return correct / difference_zero.shape[1]


def get_fold_meta_data():
    folds_path = os.path.join(workspace, 'data_folds')
    folds = ['Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
    folds_paths = []
    for i in folds:
        folds_paths.append(os.path.join(folds_path, i + '.pickle'))
    meta = []
    for i in folds_paths:
        with open(i, 'rb') as f:
            meta.append(pickle.load(f))
    meta_np = meta[0]
    for i in range(1, 4):
        meta_np = np.hstack((meta_np, meta[i]))

    return meta_np


def concat_features(features):
    dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    segments = features.shape[0]
    length = dim * segments
    temp = np.zeros((mel_bins, length))
    pointer = 0
    for i, data in enumerate(features):
        temp[:, pointer:pointer+dim] = data
        pointer += dim
    return temp


def expand_features(features):
    length = 0
    lengths = []
    features = np.squeeze(features)
    for i in features:
        split = i.shape[0] // mel_bins
        length += split
        lengths.append(split)

    reshaped = np.zeros((mel_bins, length))
    pointer = 0
    for i, data in enumerate(features):
        reshaped[:, pointer:pointer+lengths[i]] = np.reshape(data, (mel_bins,
                                                                    lengths[i]))
        pointer += lengths[i]
    return reshaped


def class_stats():
    my_logger.info(config_dataset.SEPARATOR)
    my_logger.info('Euclidean Statistics')
    meta_np = get_fold_meta_data()

    with h5py.File(data_dir, 'r') as f:
        features = f['features'][:]
    features = features[meta_np[2].tolist()]
    # features, labels = process_data(features, meta_np)
    zeros, zeros_loc, ones, ones_loc = util.count_classes(meta_np[1].tolist())

    zero_temp = features[zeros_loc]
    one_temp = features[ones_loc]

    zeros_con = expand_features(zero_temp)
    ones_con = expand_features(one_temp)

    # Perform Bin-Wise Normalisation of the data
    zero_mean = np.mean(zeros_con, axis=1)
    one_mean = np.mean(ones_con, axis=1)
    temp1 = np.sum(zeros_con**2, axis=1)
    temp2 = np.sum(ones_con**2, axis=1)
    norm = (temp1+temp2)**0.5
    zero_features_norm = zeros_con / norm.reshape(-1, 1)
    one_features_norm = ones_con / norm.reshape(-1, 1)
    zero_mean_norm = np.mean(zero_features_norm, axis=1)
    zero_std_norm = np.std(zero_features_norm, axis=1)
    one_mean_norm = np.mean(one_features_norm, axis=1)
    one_std_norm = np.std(one_features_norm, axis=1)

    my_logger.info(f"Mean of Class Zero: ")
    my_logger.info(zero_mean_norm)
    my_logger.info(f"Standard Deviation of Class Zero: ")
    my_logger.info(zero_std_norm)
    my_logger.info(f"Mean of Class One: ")
    my_logger.info(one_mean_norm)
    my_logger.info(f"Standard Deviation of Class One: ")
    my_logger.info(one_std_norm)

    # save_file = 'Non-Depressed_Vs_Depressed_Mean_Log_Mel_Features'
    # title = 'Non-Depressed Vs Depressed Mean Log Mel Features'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph(zero_mean, one_mean, save_loc, title)
    # save_file = 'Non-Depressed_Vs_Depressed_Mean_Log_Mel_Features_Normalised'
    # title = 'Non-Depressed Vs Depressed Mean Log Mel Features Normalised'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph(zero_mean_norm, one_mean_norm, save_loc, title)
    # save_file = 'Non-Depressed_Vs_Depressed_STD_Log_Mel_Features_Normalised'
    # title = 'Non-Depressed Vs Depressed STD Log Mel Features Normalised'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph(zero_std_norm, one_std_norm, save_loc, title)
    # save_file = 'Non-Depressed Mean_Std Log Mel Features_Normalised'
    # title = 'Non-Depressed Mean_Std Log Mel Features Normalised'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph_error_bars(zero_mean_norm, zero_std_norm, save_loc, title)
    # save_file = 'Depressed_Mean_Std_Log_Mel_Features_Normalised'
    # title = 'Depressed Mean_Std Log Mel Features Normalised'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph_error_bars(one_mean_norm, one_std_norm, save_loc, title)
    # save_file = \
    #     'Non-Depressed_vs_Depressed_Mean_Std_Log_Mel_Features_Normalised'
    # title = 'Non-Depressed vs Depressed Mean_Std Log Mel Features Normalised'
    # save_loc = os.path.join(stat_log_path, save_file + '.png')
    # plot_class_graph_error_bars_combined(zero_mean_norm, zero_std_norm,
    #                                      one_mean_norm, one_std_norm,
    #                                      save_loc, title)

    # euclidean_total_zero = euclidean_per_class_stats(zero_mean, zero_temp)
    # euclidean_total_one = euclidean_per_class_stats(one_mean, one_temp)
    ground_truth = 0
    correct_zeros = euclidean_stats(zero_mean, one_mean, zeros_con,
                                    ground_truth)
    print('Correct Class 0: ', correct_zeros*100, '%')
    ground_truth = 1
    correct_ones = euclidean_stats(zero_mean, one_mean, ones_con, ground_truth)
    print('Correct Class 1: ', correct_ones*100, '%')
    my_logger.info(f"Correct Classification of Class Zero: {correct_zeros}")
    my_logger.info(f"Correct Classification of Class One: {correct_ones}")

    ground_truth = 0
    correct_zeros = euclidean_stats(zero_mean, one_mean, zeros_con,
                                    ground_truth, norm.reshape(-1, 1))
    print('Correct Class 0: ', correct_zeros*100, '%')
    ground_truth = 1
    correct_ones = euclidean_stats(zero_mean, one_mean, ones_con,
                                   ground_truth, norm.reshape(-1, 1))
    print('Correct Class 1: ', correct_ones*100, '%')
    my_logger.info(f"Correct Classification of Class Zero: {correct_zeros}")
    my_logger.info(f"Correct Classification of Class One: {correct_ones}")

    ground_truth = 0
    correct_zeros = euclidean_stats_2(zero_mean, one_mean, zero_features_norm,
                                    ground_truth)
    print('Correct Class 0: ', correct_zeros*100, '%')
    ground_truth = 1
    correct_ones = euclidean_stats_2(zero_mean, one_mean, one_features_norm,
                                   ground_truth)
    print('Correct Class 1: ', correct_ones*100, '%')
    my_logger.info(f"Correct Classification of Class Zero: {correct_zeros}")
    my_logger.info(f"Correct Classification of Class One: {correct_ones}")

    euclidean_total_zero = euclidean_per_class_stats(zero_mean, features)
    euclidean_total_one = euclidean_per_class_stats(one_mean, features)

    max_z = np.max(euclidean_total_zero)
    max_o = np.max(euclidean_total_one)
    max_total = np.max(np.array([max_z, max_o]))

    euclidean_total_zero = euclidean_total_zero / max_total
    euclidean_total_one = euclidean_total_one / max_total

    fig, axs = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
    x = list(range(1, mel_bins + 1))
    axs[0].bar(x, euclidean_total_zero, label='Non-Depressed')
    axs[1].bar(x, euclidean_total_one, label='Depressed')
    fig.text(0.004, 0.5, 'Activation', va='center', rotation='vertical')
    fig.text(0.5, 0.004, 'MelBin', ha='center', rotation='horizontal')
    fig.text(0.5, 0.99, 'Histogram of Mel Bin Activation', ha='center',
             rotation='horizontal')
    axs[0].set_title('Non-Depressed', fontsize=7)
    axs[1].set_title('Depressed', fontsize=7)
    fig.subplots_adjust(top=0.88)
    # axs[0].yaxis.set_ticks(np.arange(0, 1.1, 0.05))
    axs[0].xaxis.set_ticks(np.arange(0, mel_bins+1, 5))
    plt.show()
    save_file = 'Depression Histogram Activation Log Mel Features'
    save_loc = os.path.join(stat_log_path, save_file + '.png')
    fig.savefig(save_loc)
    plt.close()

    w = 0.8
    fig, axs = plt.subplots(1, 1, figsize=(15, 15))
    axs.bar(x, euclidean_total_zero, label='Non-Depressed', color='b', width=w)
    axs.bar(x, euclidean_total_one, label='Depressed', color='r', width=0.5*w,
            alpha=0.5)
    plt.xlabel('Mel_bins')
    plt.ylabel('Activation')
    plt.title('Histogram of Depression')
    # axs.yaxis.set_ticks(np.arange(0, 1.1, 0.05))
    axs.xaxis.set_ticks(np.arange(0, mel_bins+1, 5))
    plt.legend()
    save_file = 'Depression_Histogram_Activation_Log_Mel_Features_Combined'
    save_loc = os.path.join(stat_log_path, save_file + '.png')
    fig.savefig(save_loc)
    plt.show()


if __name__ == '__main__':
    config = importlib.import_module('config')
    start_time = time.time()
    workspace = config.WORKSPACE_MAIN_DIR
    folder_name = config.FOLDER_NAME
    my_logger, stat_log_path = setup()
    mel_bins = config.EXPERIMENT_DETAILS['MEL_BINS']
    data_dir = os.path.join(workspace, folder_name, 'complete_database.h5')
    data_folds = config.EXPERIMENT_DETAILS['TOTAL_FOLDS']
    # fold_stats()
    class_stats()
    print(f"Complete Runtime: {time.time()-start_time}")
