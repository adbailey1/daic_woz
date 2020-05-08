import os
import pickle
import numpy as np
import h5py
import pandas as pd
import argparse
import logging
import logging.handlers
import csv
import shutil
import torch
import random


def save_model(epoch_iter, model, optimizer, main_logger, model_dir,
               data_saver):
    """
    Saves the model weights along with the current epoch and all the random
    states that are used during the experiment. Also saves the current state
    of the data loader for continuity

    Inputs:
        epoch_iter: Current epoch
        model: The model from current experiment
        optimizer: The current optimiser
        main_logger: The logger used for recording important information
        model_dir: Location of the model to be saved
        data_saver: Holds information regarding the data loader so that it
                    can be restored from a checkpoint. This includes the
                    current pointer of ones and zeros and the current list of
                    indexes of the ones and zeros
    """
    print('Saving the Model at epoch: ', epoch_iter)
    main_logger.info(f"Saving the model at epoch_iter: {epoch_iter}")
    save_out_dict = {'epoch': epoch_iter,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'rng_state': torch.get_rng_state(),
                     'cuda_rng_state': torch.cuda.get_rng_state(),
                     'numpy_rng_state': np.random.get_state(),
                     'random_rng_state': random.getstate()
                     }
    save_out_path = os.path.join(model_dir,
                                 f"md_{epoch_iter}_epochs.pth")
    torch.save(save_out_dict, save_out_path)

    save_out_path = os.path.join(model_dir, 'data_saver.pickle')
    with open(save_out_path, 'wb') as f:
        pickle.dump(data_saver, f)


def load_model(checkpoint_path, model, optimizer):
    """
    Loads the model weights along with the current epoch and all the random
    states that are used during the experiment. Also loads the current state
    of the data loader for continuity

    Inputs:
        checkpoint_path: Location of the saved model
        model: The model from current experiment
        optimizer: The current optimiser state

    Outputs:
        epoch_iter: Current epoch
        data_saver: Holds information regarding the data loader so that it
            can be restored from a checkpoint. This includes the
            current pointer of ones and zeros and the current list of
            indexes of the ones and zeros

    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_rng_state'])

    data_saver_path = checkpoint_path.split('/')
    data_saver_path = checkpoint_path.replace(data_saver_path[-1],
                                              'data_saver.pickle')
    with open(data_saver_path, 'rb') as f:
        data_saver = pickle.load(f)

    return epoch, data_saver


def save_model_outputs(model_dir, dataframe, train_pred, val_pred,
                       best_scores, best_scores_2):
    """
    Saves the outputs of a model for checkpointing or future analysis for a
    completed experiment.

    Input
        model_dir: Location of the data to be saved
        dataframe: pandas dataframe containing the results at each epoch up
                   to the checkpoint
        train_pred: Outputs of the training batches at each epoch up to the
                    checkpoint
        val_pred: Outputs of the validation batches at each epoch up to the
                  checkpoint
        best_scores: Record of the best performing iteration of the model
        best_scores_2: Same as best_scores but more accurate and only
                       holds validation position
    """
    save_path = os.path.join(model_dir, 'complete_results.pickle')
    dataframe.to_pickle(save_path)

    save_path = os.path.join(model_dir, 'predicted_labels_train_val.pickle')
    complete_predictions = [train_pred, val_pred]
    with open(save_path, 'wb') as f:
        pickle.dump(complete_predictions, f)

    save_path = os.path.join(model_dir, 'best_scores.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump((best_scores, best_scores_2[1:]), f)


def load_model_outputs(model_dir):
    """
    Loads the saved outputs of a model from a checkpoint.

    Input
        model_dir: Location of the data to be loaded

    Outputs:
        dataframe: pandas dataframe containing the results at each epoch up
                   to the checkpoint
        train_pred: Outputs of the training batches at each epoch up to the
                    checkpoint
        val_pred: Outputs of the validation batches at each epoch up to the
                  checkpoint
        best_scores: Record of the best performing iteration of the model
        best_scores_2: Same as best_scores but more accurate and only
                       holds validation position
    """
    load_path = os.path.join(model_dir, 'complete_results.pickle')
    with open(load_path, 'rb') as f:
        dataframe = pickle.load(f)

    load_path = os.path.join(model_dir, 'predicted_labels_train_val.pickle')
    with open(load_path, 'rb') as f:
        complete_predictions = pickle.load(f)
    train_pred, val_pred = complete_predictions

    load_path = os.path.join(model_dir, 'best_scores.pickle')
    with open(load_path, 'rb') as f:
        best_scores, best_scores_2 = pickle.load(f)

    return dataframe, train_pred, val_pred, best_scores, best_scores_2


def create_directories(location, folders_to_make):
    """
    Creates a directory (and potential sub directories) at a location

    Input
        location: location of the new directories
        folders_to_make: List of the sub directories
    """

    for i in folders_to_make:
        os.mkdir(os.path.join(location, i))


def get_labels_from_dataframe(path):
    """
    Reads database labels from csv file using pandas.

    Input
        path: The location of the database labels csv file

    Output:
        output: List containing the Participant IDs and the classes/scores
    """
    df = pd.read_csv(path)
    output = [df['Participant_ID'].values.tolist(),
              df['PHQ8_Binary'].values.tolist(),
              df['PHQ8_Score'].values.tolist()]

    return output


def seconds_to_sample(seconds, window_size, overlap=0, hop_length=0,
                      sample_rate=16000, feature_type='logmel'):
    """
    Converts number of seconds into the equivalent number of samples taking
    into account the type of feature. For example raw audio will simply be
    the seconds * sample rate whereas logmel will require further calculation
    as the process of creating logmel compresses the data along the time axis

    Inputs:
        seconds: Number of seconds to convert
        window_size: Length of window used in feature extraction of logmel
                     for example
        overlap: Overlap used in feature extraction for logmel for example
        hop_length: Hop length used in feature extraction of logmel for example
        sample_rate: Original sampling rate of the data
        feature_type: What type of feature is used? Raw audio? Logmel?

    Outputs:
        samples: Converted samples
    """
    if overlap == 0 and hop_length == 0:
        hop_length = window_size // 2
    elif hop_length == 0 and overlap != 0:
        overlap = overlap / 100
        overlap = window_size * overlap
        hop_length = window_size - round(overlap)

    num_sample = seconds * sample_rate
    if feature_type == 'raw':
        samples = int(num_sample)
    else:
        num_sample = num_sample - (window_size/2)
        num_sample = num_sample // hop_length
        samples = int(num_sample + 2)

    return samples


def count_classes(complete_classes):
    """
    Counts the number of zeros and ones in the dataset:

    Input:
        complete_classes: List of the classes of the dataset

    Outputs:
        zeros: Number of zeros in the dataset
        index_zeros: Indexes of the zeros in the dataset w.r.t. feature array
        ones: Number of ones in the dataset
        index_ones: Indexes of the ones in the dataset w.r.t. feature array
    """
    zeros = 0
    ones = 0
    index_zeros = []
    index_ones = []
    for i, d in enumerate(complete_classes):
        if d == 0:
            zeros += 1
            index_zeros.append(i)
        else:
            ones += 1
            index_ones.append(i)

    return zeros, index_zeros, ones, index_ones


def load_data(path, labels):
    """
    Loads specific data from a dataset using indexes from labels.

    Input:
        path: The location to the database
        labels: The database labels which include the indexes of the specific
                data to load

    Output:
        features: The dataset features
    """
    with h5py.File(path, 'r') as h5:
        features = h5['features'][:]

    features = features[labels[-1].tolist()]

    return features


def load_labels(path):
    """
    Loads the labels for a dataset at a given location.

    Input:
        path: The location to the database labels

    Output:
        labels: The labels for the dataset
    """
    if isinstance(path, list):
        for i, file in enumerate(path):
            with open(file, 'rb') as f:
                if i == 0:
                    labels = pickle.load(f)
                else:
                    labels = np.concatenate((labels,
                                             pickle.load(f)),
                                            axis=1)
    else:
        with open(path, 'rb') as f:
            labels = pickle.load(f)

    return labels


def str2bool(arg_value):
    """
    When parsing in boolean values, for some reason argparse doesn't register
    the initial values, therefore it will always output as True, even if they
    are parsed in as False. This function is used in place of the type
    argument in the argparse.add_argument and fixes this issue. From
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with
    -argparse

    Input
        arg_value: Value parsed in as an argument

    """

    if isinstance(arg_value, bool):
        return arg_value
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logger(current_directory):
    """
    Setup the logger for the current experiment

    Input
        current_directory: The location of the logger to be stored

    Output
        main_logger: The logger to be used throughout the experiment
    """
    log_path = os.path.join(current_directory, 'audio_file_analysis.log')
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)

    return main_logger


def csv_read(file, start=None, end=None):
    """
    Read a csv (comma separated value) file and append each line to a list

    Input:
        file: The location of the csv file
        start: Start location for a read line
        end: End location for a read line

    Output:
        data: List of each row from csv file
    """
    data = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if start is not None and end is not None:
                    data.append(row[start:end])
                else:
                    data.append(row)
    return data


def remove_directory(location):
    """
    Removes a directory and all sub directories at a specific location

    Input:
        location: Location of the directory to be removed
    """
    shutil.rmtree(location, ignore_errors=False, onerror=None)


def normalise(data, mean, std):
    """
    From a set of data, normalise the data using the mean and the standard
    deviation to obtain 0 mean and standard deviation of 1

    Inputs:
        data: The data to be processed
        mean: The mean of the data
        std: The standard deviation of the data

    Output:
        normalised_data: Output normalised data with mean 0 and standard
                         deviation of 1
    """
    normalised_data = (data-mean) / std

    return normalised_data
