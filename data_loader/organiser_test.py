import os
import pickle
import torch
import numpy as np
from utils import utilities as util
from data_loader import data_gen
from exp_run import audio_feature_extractor as afe
from scipy import interpolate
import cv2


def get_updated_features(data, min_samples, label, meta, logger, segment_dim,
                         freq_bins, amcs, convert_to_image, initialiser,
                         feature_exp):
    """
    Final stage before segmenting the data into the form N, F, S where N is
    number of new dimensions related to the value of segment_dim. If
    'whole_train' is selected, interpolation will be applied. If the shorten
    dataset option is provided, the files will be shortened before processing

    Inputs:
        data: Array of the data in the database
        min_samples: The size of the shortest file in the database
        label: Set for 'train', 'dev', or 'test'
        meta: Meta data including folder, class, score and original index
        logger: The main logger for recording important information
        segment_dim: The segmentation dimensions for the updated data
        freq_bins: Number of bins for the feature. eg logmel - 64
        amcs: Bool - Audio Mode is Concat not Shorten
        convert_to_im: Bool - will data be converted to 3D?
        initialiser: Used to log information on the first iteration of outer
                     loop
        training: Determines the data processing, 'random_sample',
                  'chunked_file', or 'whole_file'
        max_value: The longest file in the dataset used for 'whole_file'
        feature_exp: Type of features used in this experiment, eg. logmel

    Outputs:
        new_features: Updated array of features N, F, S where S is the
                      feature dimension specified in the config file.
        new_folders: Updated list of folders
        new_classes: Updated list of classes
        new_scores: Updated list of scores
        new_indices: Updated list of indices
    """
    dimension = data.shape[0] // freq_bins
    reshaped_data = np.reshape(data, (freq_bins, dimension))
    if not amcs and label != 'dev':
        reshaped_data = reshaped_data[:, 0:min_samples]
        if initialiser == 0:
            logger.info(f"The {label} data has been shortened to "
                        f"{reshaped_data.shape[0]} by"
                        f" {reshaped_data.shape[1]}")

    new_meta_data = afe.feature_segmenter(reshaped_data, meta, feature_exp,
                                          segment_dim, convert_to_image)
    return new_meta_data


def calculate_length(data, min_samples, label, segment_dim, freq_bins,
                     amcs=True):
    """
    Calculates the length of the updated array once the features have been
    segmented into dimensions specified by segment_dimension in config file

    Inputs:
        data: The data to be segmented
        min_samples: This is the shortest file in the dataset
        label: Set to 'train', 'dev', or 'test'
        segment_dim: Dimensions to reshape the data
        freq_bins: The number of bins used, eg. logmel = 64
        amcs: Bool - Audio Mode is Concat not Shorten

    Output:
        length: The length of the dimension of the data after segmentation
    """
    dimension = data.shape[0] // freq_bins
    if amcs or label == 'dev':
        if dimension % segment_dim == 0:
            length = (dimension // segment_dim)
        else:
            length = (dimension // segment_dim) + 1
    else:
        if min_samples % segment_dim == 0:
            length = (min_samples // segment_dim)
        else:
            length = (min_samples // segment_dim) + 1

    return length


def process_data(amcs, freq_bins, features, labels, mode_label, min_samples,
                 logger, segment_dim, convert_to_im, feature_exp):
    """
    Determine the array size of the dataset once it has been reshaped into
    segments of length equal to that specified in the config file for feature
    dimensions. Following this, create the arrays with respect to the chosen
    feature type of the data. Update the folders, class, score and index lists.

    Inputs:
        amcs: Bool - Audio Mode is Concat not Shorten
        freq_bins: Number of bins for the feature. eg logmel - 64
        features: Array of the features in the database
        labels; Labels corresponding to the features in the database
        mode_label: Set for 'train', 'dev', or 'test'
        min_samples: The size of the shortest file in the database
        logger: The main logger for recording important information
        segment_dim: The segmentation dimensions for the updated data
        convert_to_im: Bool - will data be converted to 3D?
        training: Determines the data processing, 'random_sample',
                  'chunked_file', or 'whole_file'
        max_value: The longest file in the dataset used for 'whole_file'
        feature_exp: Type of features used in this experiment, eg. logmel

    Outputs:
        update_features: Updated array of features N, F, S where S is the
                         feature dimension specified in the config file.
        update_labels: Updated lists containing the folders, classes, scores,
                       and indices after segmentation
        locator: List of the length of every segmented data
    """
    # Work out how many dimensions the segmented feature dataset will have
    length = 0
    for pointer, i in enumerate(labels[0, :]):
        i = features[pointer, 0]
        temp_length = calculate_length(i, min_samples, mode_label,
                                       segment_dim, freq_bins, amcs)

        if not amcs and mode_label != 'dev':
            length = temp_length * features.shape[0]
            break
        length += temp_length
    if convert_to_im:
        if feature_exp == 'MFCC_concat':
            update_features = np.zeros((length, freq_bins*3,
                                        segment_dim),
                                       dtype=np.float32)
        else:
            update_features = np.zeros((length, 3, freq_bins,
                                        segment_dim),
                                       dtype=np.float32)
    else:
        update_features = np.zeros((length, freq_bins, segment_dim),
                                   dtype=np.float32)

    pointer = 0
    locator = []
    final_folders = []
    final_classes = []
    final_scores = []
    final_genders = []
    final_indices = []
    initialiser = 0
    for i, data in enumerate(labels[0, :]):
        data = features[i, 0]
        # Folder, class, score, gender
        meta = [labels[0][i], labels[1][i], labels[2][i], labels[3][i]]
        (new_features, new_folders, new_classes, new_scores, new_genders,
         new_indices) = get_updated_features(data, min_samples, mode_label,
                                             meta, logger, segment_dim,
                                             freq_bins, amcs, convert_to_im,
                                             initialiser, feature_exp)
        initialiser += 1

        if convert_to_im:
            if feature_exp == 'MFCC_concat':
                z, _, _ = new_features.shape
                update_features[pointer:pointer + z, :, :] = new_features
            else:
                z, _, _, _ = new_features.shape
                update_features[pointer:pointer + z, :, :, :] = new_features
        else:
            z, _, _ = new_features.shape
            update_features[pointer:pointer + z, :, :] = new_features
        locator.append([pointer, pointer+z])
        new_indices = new_indices + pointer
        final_folders.append(new_folders)
        final_classes.append(new_classes)
        final_scores.append(new_scores)
        final_genders.append(new_genders)
        if type(new_indices) is int:
            final_indices.append(new_indices)
        else:
            final_indices.append(new_indices.tolist())
        pointer += z

    print(f"The dimensions of the {mode_label} features are:"
          f" {update_features.shape}")
    logger.info(f"The dimensions of the {mode_label} features are:"
                f" {update_features.shape}")
    if type(final_folders[0]) is list:
        final_folders = [j for i in final_folders for j in i]
    if type(final_classes[0]) is list:
        final_classes = [j for i in final_classes for j in i]
    if type(final_scores[0]) is list:
        final_scores = [j for i in final_scores for j in i]
    if type(final_genders[0]) is list:
        final_genders = [j for i in final_genders for j in i]
    if type(final_indices[0]) is list:
        final_indices = [j for i in final_indices for j in i]
    update_labels = [final_folders, final_classes, final_scores,
                     final_genders, final_indices]

    return update_features, update_labels, locator


def determine_folds(current_fold, total_folds):
    """
    Function to determine which folds will be used for training and which
    will be held out for validation

    Inputs:
        current_fold: Current fold of the experiment (will be used as
                      validation fold)
        total_folds: How many folds will this dataset have?

    Outputs:
        folds_for_train: List of training folds
        folds_for_dev: List of validation folds
    """
    folds_for_train = []
    for fold_num in list(range(1, total_folds+1)):
        if current_fold == fold_num:
            folds_for_dev = current_fold
        else:
            folds_for_train.append(fold_num)
    return folds_for_train, folds_for_dev


def file_paths(features_dir, config, logger, current_fold, data_fold_dir,
               data_fold_dir_equal):
    """
    Determines the file paths for the training fold data, validation fold
    data, the summary file created in the data processing stage, and the
    database.

    Inputs:
        features_dir: Directory of the created features
        config: Config file to be used for this experiment
        logger: For logging relevant information
        current_fold: This is used to hold out a specific fold for validation

    Outputs:
        train_meta_file: File paths for the training folds
        dev_meta_file: File path for the validation fold
        sum_file: File path of the summary file (holds meta information)
        h5_database: File path of the database
    """
    make_dataset_equal = config.EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']
    total_folds = config.EXPERIMENT_DETAILS['TOTAL_FOLDS']

    if make_dataset_equal:
        folder = data_fold_dir_equal
    else:
        folder = data_fold_dir

    folds_for_train, folds_for_dev = determine_folds(current_fold, total_folds)
    logger.info(f"Training Folds: {folds_for_train}, "
                f"Dev Folds: {folds_for_dev}")

    dev_meta_file = os.path.join(features_dir, data_fold_dir,
                                 f"Fold_{folds_for_dev}.pickle")
    test_meta_file = os.path.join(features_dir, data_fold_dir, 'test.pickle')
    sum_file = os.path.join(features_dir, 'summary.pickle')
    if config.GENDER == 'm' or config.GENDER == 'f':
        h5_database = os.path.join(features_dir,
                                   'complete_database'+'_'+config.GENDER+'.h5')
    else:
        h5_database = os.path.join(features_dir, 'complete_database.h5')

    return dev_meta_file, test_meta_file, sum_file, h5_database


def find_weight(zeros, ones):
    """
    Finds the balance of the dataset to create weights for training

    Inputs:
        zeros: Number of zeros in the database
        ones: Number of ones in the database
        activation: Is the final layer sigmoid function or softmax function?
        batch_size: The batch size to be used during training

    Output:
        weights: Array specifying the weights for the current dataset.
    """
    min_class = min(zeros, ones)
    max_class = max(zeros, ones)
    class_weight = min_class / max_class
    index_to_alter = [zeros, ones].index(max_class)

    weights = [1, 1]
    weights[index_to_alter] = class_weight

    return weights


def data_info(labels, value, logger, config):
    """
    Log the number of ones and zeros in the current set. If class_weights is
    selected, determine the balance of the dataset

    Inputs:
        labels: The labels for the current set of data
        value: Set to training or validation
        logger: To record important information
        config: Config file for state information

    Outputs:
        zeros: Number of zeros in the current set
        zeros_index: Indices of the zeros of the set w.r.t. feature array
        ones: Number of ones in the current set
        ones_index: Indices of the ones of the set w.r.t. feature array
        class_weights: The class weights for current set
    """
    zeros, zeros_index, ones, ones_index = util.count_classes(labels[1])

    print(f"The number of class zero and one files in the {value} split after "
          f"segmentation are {zeros}, {ones}")
    logger.info(f"The number of class zero files in the {value} split "
                f"after segmentation are {zeros}")
    logger.info(f"The number of class one files in the {value} split "
                f"after segmentation are {ones}")

    use_class_weights = config.EXPERIMENT_DETAILS['CLASS_WEIGHTS']
    if use_class_weights:
        class_weights = find_weight(zeros, ones)
    else:
        class_weights = [1, 1]

    class_weights = torch.Tensor(class_weights)

    logger.info(f"Class Weights: {class_weights}")

    return zeros, zeros_index, ones, ones_index, class_weights


def group_data(features, feat_shape, feat_dim, freq_bins,
               convert_to_im):
    """
    Function to split features into a set determined by feat_dim which will
    be used in batches for training. For example, if features.shape = [10,
    64, 100] and feat_dim = 20, each data will be split into 5 (100/20). As
    there are 10 data, the new dimensions will be 5*10
    Inputs:
        features: Features used for training
        feat_shape: Features.shape
        feat_dim: Dimensions to be used for batches
        freq_bins: Number of bins could me Mel or otherwise
        convert_to_im: Bool, will the features be converted into 3D

    Outputs:
        updated_features: Features split into batch form
        updated_locator: List detailing the length of each data after updating
    """
    if feat_shape[-1] % feat_dim == 0:
        new_dim = (feat_shape[-1] // feat_dim)
    else:
        new_dim = (feat_shape[-1] // feat_dim) + 1
    new_dim2 = new_dim * feat_shape[0]
    if convert_to_im:
        updated_features = np.zeros((new_dim2, 3, freq_bins, feat_dim),
                                    dtype=np.float32)
    else:
        updated_features = np.zeros((new_dim2, freq_bins, feat_dim),
                                    dtype=np.float32)
    pointer = 0
    updated_locator = []
    for i in features:
        last_dim = i.shape[-1]
        if last_dim % feat_dim == 0:
            leftover = 0
        else:
            leftover = feat_dim - (last_dim % feat_dim)
        if convert_to_im:
            i = np.dstack((i, np.zeros((3, freq_bins, leftover))))
            updated_features[pointer:pointer + new_dim, :, :, :] = np.split(i,
                                                                            new_dim,
                                                                            axis=-1)
        else:
            i = np.hstack((i, np.zeros((freq_bins, leftover))))
            updated_features[pointer:pointer + new_dim, :, :] = np.split(i,
                                                                         new_dim,
                                                                         axis=1)
        updated_locator.append([pointer, pointer + new_dim])
        pointer += new_dim

    return updated_features, updated_locator


def determine_seconds_segment(seconds_segment, feature_dim, window_size, hop,
                              learning_proc, feat_type):
    """
    Determines the number of samples for a given number of seconds of audio
    data. For example if the sampling rate is 16kHz and the data should be
    clustered to 30s chunks then it will have time dimensionality 16k * 30 =
    480000 samples. However, if the data is in the form of mel bin for
    example, the data has already been processed by a window function to
    compress the data along the time axis and so this must be taken into
    account.

    Inputs:
        seconds_segment: The number of seconds the user wants to cluster
        feature_dim: Number of samples for batching
        window_size: Window size used in cases of logmel for example
        hop: Hop length in cases of logmel for example
        learning_proc: How to process the data, random_sample (each sample
                       length determined by feature_dim), chunked_file (length
                       determined by seconds_segment), or whole_file
        feat_type: What type of audio data are we using? Raw? Logmel?

    Output:
        seconds_segment: Updated in terms of samples rather than seconds
    """
    if learning_proc == 'chunked_file':
        seconds_segment = util.seconds_to_sample(seconds_segment,
                                                 window_size=window_size,
                                                 hop_length=hop,
                                                 feature_type=feat_type)
    elif learning_proc == 'random_sample' or learning_proc == 'whole_file':
        seconds_segment = feature_dim

    return seconds_segment


def organise_data(config, logger, file, database, min_samples,
                  learning_procedure, mode_label='train'):
    amcs = config.EXPERIMENT_DETAILS['AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    feature_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    convert_to_im = config.EXPERIMENT_DETAILS['CONVERT_TO_IMAGE']
    feature_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    window_size = config.WINDOW_SIZE
    hop = config.HOP_SIZE
    labels = util.load_labels(file)
    features = util.load_data(database, labels)

    seconds_segment = determine_seconds_segment(
        config.EXPERIMENT_DETAILS['SECONDS_TO_SEGMENT'], feature_dim,
        window_size, hop, learning_procedure, feature_exp)

    features, labels, loc = process_data(amcs, freq_bins, features, labels,
                                         mode_label, min_samples, logger,
                                         seconds_segment, convert_to_im,
                                         feature_exp)

    if learning_procedure == 'chunked_file':
        feat_shape = features.shape
        features, loc = group_data(features, feat_shape, feature_dim, freq_bins,
                                   convert_to_im)

    index = [0, 0]
    class_weights = [1, 1]
    zeros = ones = 0
    if mode_label == 'dev':
        _, zero_index, _, one_index, class_weights = data_info(labels,
                                                               mode_label,
                                                               logger, config)
        index = [zero_index, one_index]

    return features, labels, index, loc, (zeros, ones, class_weights)


def gender_split_indices(data):
    if isinstance(data, list):
        data = np.array(data)
    male_dep_indices = list(np.where((data[1, :] == 1) & (data[-2, :] == 1))[0])
    male_ndep_indices = list(np.where((data[1, :] == 0) & (data[-2, :] ==
                                                           1))[0])
    female_dep_indices = list(np.where((data[1, :] == 1) & (data[-2, :] ==
                                                           0))[0])
    female_ndep_indices = list(np.where((data[1, :] == 0) & (data[-2, :] ==
                                                           0))[0])

    male_dep = len(male_dep_indices)
    male_ndep = len(male_ndep_indices)
    fem_dep = len(female_dep_indices)
    fem_ndep = len(female_ndep_indices)

    min_value = min([male_dep, male_ndep, fem_dep, fem_ndep])

    fem_nd_w = find_weight(fem_ndep, min_value)
    fem_d_w = find_weight(fem_dep, min_value)
    male_nd_w = find_weight(male_ndep, min_value)
    male_d_w = find_weight(male_dep, min_value)
    weights = (torch.Tensor(fem_nd_w), torch.Tensor(fem_d_w),
               torch.Tensor(male_nd_w), torch.Tensor(male_d_w))

    return female_ndep_indices, female_dep_indices, male_ndep_indices, \
           male_dep_indices, weights


def run(config, logger, current_fold, checkpoint, data_fold_dir,
        data_fold_dir_equal, features_dir, data_saver, tester=False):
    """
    High level function to process the training and validation data. This
    function obtains the file locations, folds for training/validation sets,
    processes the data to be used for training.
    Inputs:
        config: Config file holding state information for experiment
        logger: Logger to record important information
        current_fold: Current fold for experiment to be used to determine the
                      training and validation folds
        checkpoint: Is there a checkpoint to load from?

    Outputs:
        generator: Generator for training and validation batch data loading
        class_weights_train:
        class_weights_dev
        zero_train
        one_train
    """
    feature_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    learning_procedure = config.LEARNING_PROCEDURE_DEV

    dev_file, test_file, summary_file, database = file_paths(features_dir,
                                                             config, logger,
                                                             current_fold,
                                                             data_fold_dir,
                                                             data_fold_dir_equal)
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    logger.info(f"The dimensions of the logmel features before segmentation "
                f"are: {summary[1][-1]}")
    if feature_exp == 'raw':
        min_samples = int(summary[1][summary[0].index('MinSamples')])
    else:
        min_samples = int(summary[1][summary[0].index('MinWindows')])

    if tester:
        features, labels, index, loc, class_data = organise_data(config, logger,
                                                                 test_file,
                                                                 database,
                                                                 min_samples,
                                                                 learning_procedure,
                                                                 mode_label='test')
    else:
        features, labels, index, loc, class_data = organise_data(config,
                                                                 logger,
                                                                 dev_file,
                                                                 database,
                                                                 min_samples,
                                                                 learning_procedure,
                                                                 mode_label='dev')

    gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    class_weights = data_saver['class_weights']
    if gender_balance and not tester:
        # Gender weights tuple of (fem_nd_w, fem_d_w, male_nd_w, male_d_w)
        female_ndep_ind, female_dep_ind, male_ndep_ind, male_dep_ind, \
        gender_weights = gender_split_indices(labels)
        index = [female_ndep_ind, female_dep_ind, male_ndep_ind,
                 male_dep_ind]
        for p, i in enumerate(labels[3]):
            if labels[3][p] == 0 and i == 0:
                pass
            elif labels[3][p] == 0 and i == 1:
                labels[3][p] = 2
            elif labels[3][p] == 1 and i == 0:
                labels[3][p] = 1
            else:
                labels[3][p] = 3

    generator = data_gen.GenerateData(train_labels=None,
                                      dev_labels=labels,
                                      train_feat=None,
                                      dev_feat=features,
                                      train_loc=None,
                                      dev_loc=loc,
                                      train_indices=index,
                                      dev_indices=index,
                                      logger=logger,
                                      config=config,
                                      checkpoint=checkpoint,
                                      gender_balance=gender_balance,
                                      data_saver=data_saver)

    return generator, (0, 0, class_weights)
