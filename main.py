import logging
import logging.handlers
import argparse
import os
import time
from models_pytorch import Custom, CustomAtt
from data_loader import organiser
import shutil
from shutil import copyfile
import sys
import torch
import random
import numpy as np
import pandas as pd
import dataset_processing
import utilities as util
import socket
import importlib
import utilities
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import natsort
import model_utilities as mu
from plotter import plot_graph, confusion_mat
import config_dataset

EPS = 1e-12


def calculate_accuracy(target, predict, classes_num, f_score_average):
    """
    Calculates accuracy, precision, recall, F1-Score, True Negative,
    False Negative, True Positive, and False Positives of the output of
    the model

    Inputs
      target: np.array() The labels for the predicted outputs from the model
      predict: np.array() The batched outputs of the network
      classes_num: int How many classes are in the dataset
      f_score_average: str How to average the F1-Score

    Outputs:
      accuracy: Float Accuracy of the model outputs
      p_r_f: Array of Floats Precision, Recall, and F1-Score
      tn_fp_fn_tp: Array of Floats True Negative, False Positive,
                   False Negative, and True Positive
    """

    number_samples_labels = len(target)

    number_correct_predictions = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(number_samples_labels):
        total[target[n]] += 1

        if target[n] == predict[n]:
            number_correct_predictions[target[n]] += 1

    con_matrix = confusion_matrix(target, predict)
    # print(con_matrix)
    cm = confusion_mat(target, predict)
    # print(cm)
    tn_fp_fn_tp = con_matrix.ravel()
    if tn_fp_fn_tp.shape != (4,):
        value = int(tn_fp_fn_tp)
        if target[0][0] == 1:
            tn_fp_fn_tp = np.array([0, 0, 0, value])
        elif target[0][0] == 0:
            tn_fp_fn_tp = np.array([value, 0, 0, 0])
        else:
            print('Error in the true_neg/false_pos value')
            sys.exit()

    if f_score_average is None:
        # This code fixes the divide by zero error
        accuracy = np.divide(number_correct_predictions,
                             total,
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                          predict)
    elif f_score_average == 'macro':
        # This code fixes the divide by zero error
        accuracy = np.divide(number_correct_predictions,
                             total,
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                          predict,
                                                          average='macro')
    elif f_score_average == 'micro':
        # This code fixes the divide by zero error
        accuracy = np.divide(np.sum(number_correct_predictions),
                             np.sum(total),
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                          predict,
                                                          average='micro')
    else:
        raise Exception('Incorrect average!')

    if p_r_f[0].shape == (1,):
        temp = np.zeros((4, 2))
        position = int(target[0])
        for val in range(len(p_r_f)):
            temp[val][position] = float(p_r_f[val])

        temp1 = (temp[0])
        temp2 = (temp[1])
        temp3 = (temp[2])
        temp4 = (temp[3])
        p_r_f = (temp1, temp2, temp3, temp4)

    return accuracy, p_r_f, tn_fp_fn_tp


def forward(generate_dev, return_target, net_params, recurrent_out,
            convert_image):
    """
    Pushes the data to the model and collates the outputs

    Inputs:
        generate_dev: generator - holds the batches for the validation
        return_target: Bool - If True, return labels and folders information
        net_params: dictionary - Holds the model configurations
        recurrent_out: str - Type of output of RNN layer


    Output:
        results_dict: dictionary - Outputs, optional - labels and folders
    """
    outputs = []
    if return_target:
        targets = []
        folders = []
    # Evaluate on mini-batch
    for data in generate_dev:
        if return_target:
            (batch_data, batch_label, batch_folder, batch_locator) = data
        else:
            batch_data = data
        # Predict
        model.eval()
        # Potentially speeds up evaluation and memory usage
        with torch.no_grad():
            batch_output = get_output_from_model(data=batch_data,
                                                 net_type=select_net,
                                                 net_params=net_params,
                                                 learning_procedure=learning_procedure_dev,
                                                 learning_procedure_decider=learning_procedure_decider_dev,
                                                 locator=batch_locator,
                                                 label='dev',
                                                 recurrent_out=recurrent_out,
                                                 convert_to_image=convert_image)
        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        if return_target:
            targets.append(batch_label)
            folders.append(batch_folder)

    results_dict = {}

    outputs = np.concatenate(outputs, axis=0)
    results_dict['output'] = outputs
    if return_target:
        targets = np.concatenate(targets, axis=0)
        results_dict['target'] = np.array(targets)
        folders = np.concatenate(folders, axis=0)
        results_dict['folder'] = np.array(folders)

    return results_dict


def evaluate(generator, data_type, class_weights, comp_res, class_num,
             f_score_average, averaging, net_p, recurrent_out, epochs,
             convert_im):
    """
    Processes the validation set by creating batches, passing these through
    the model and then calculating the resulting loss and accuracy metrics.

    Input
        generator: generator - Created to load validation data to the model
        data_type: str - set to 'dev' or 'test'
        class_weights: tensor - if class weights are used due to imbalanced
                       dataset
        comp_res: dataframe - holds the complete results for training and
                  validation up to the current epoch
        class_num: int - Number of classes in the dataset
        f_score_average: str - Type of F1 Score processing
        averaging: str - Geometric or arithmetic for the outputs from the model
        net_p: dictionary - holds the model configurations
        recurrent_out: str - If RNN used, how to process the output
        epochs: int - The current epoch

    Returns:
        complete_results: dataframe - Updated dataframe of results
        per_epoch_pred: numpy.array - collated outputs and labels from the
                        current validation test
    """

    # Generate function
    print('Generating data for evaluation')
    generate_dev = generator.generate_development_data(epoch=epochs)
    # Forward
    start_time = time.time()
    return_target = True
    results_dict = forward(generate_dev=generate_dev,
                           return_target=return_target,
                           net_params=net_p,
                           recurrent_out=recurrent_out,
                           convert_image=convert_im)
    end_time = time.time()
    calc_time = end_time - start_time
    print(f"It took: {calc_time:.2f}s to process {data_type}, in evaluate mode")
    main_logger.info(f"It took: {calc_time:.2f}s to process {data_type}, "
                     f"in evaluate mode")
    outputs = results_dict['output']  # (audios_num, classes_num)
    if return_target:
        targets = results_dict['target']  # (audios_num, classes_num)
        folders = results_dict['folder']

    if learning_procedure_decider_dev != 'whole_file':
        collected_output = {}
        new_targets = []
        counter = {}
        for p, fol in enumerate(folders):
            if fol not in collected_output.keys():
                # collected_output[fol] = [outputs[p][0]]
                new_targets.append(targets[p])
                collected_output[fol] = outputs[p]
                counter[fol] = 1
            else:
                # collected_output[fol].append(outputs[p][0])
                collected_output[fol] += outputs[p]
                counter[fol] += 1

        new_outputs = []
        for co in collected_output:
            temp = collected_output[co] / counter[co]
            # mean_val = sum(temp) / len(temp)
            new_outputs.append(temp)

        outputs = np.array(new_outputs)
        targets = np.array(new_targets)

    loss = mu.calculate_loss(torch.Tensor(outputs),
                             torch.LongTensor(targets),
                             averaging,
                             class_weights,
                             net_p)

    complete_results, per_epoch_pred = prediction_and_accuracy(outputs,
                                                               targets,
                                                               True,
                                                               class_num,
                                                               comp_res,
                                                               loss, 0,
                                                               config,
                                                               f_score_average)

    return complete_results, per_epoch_pred


def logging_info():
    """
    Sets up the logger to be used for the current experiment. This is useful
    to capture relevant information during the course of the experiment.

    Output
        main_logger: logger - The created logger
    """
    log_path = os.path.join(current_dir, 'log', f"model_{current_fold}.log")
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)

    main_logger.info(config_dataset.SEPARATOR)
    main_logger.info('EXPERIMENT DETAILS')
    for dict_val in config.EXPERIMENT_DETAILS:
        if dict_val == 'SEED':
            main_logger.info(f"Starting {dict_val}:"
                             f" {str(config.EXPERIMENT_DETAILS[dict_val])}")
        else:
            main_logger.info(f"{dict_val}:"
                             f" {str(config.EXPERIMENT_DETAILS[dict_val])}")
    main_logger.info(f"Current Seed: {chosen_seed}")
    main_logger.info(f"Logged into: {socket.gethostname()}")
    main_logger.info('Experiment details: 1 conv: 2 FC: No-Oversampling/Class_weights')
    main_logger.info(config_dataset.SEPARATOR)

    return main_logger


def create_model(main_logger):
    """
    Creates the model to be used in the current experimentation

    Input
        main_logger: logger - Used to capture important information

    Output
        model: obj - The model to be used for training during experiment
    """
    if select_net == 'custom':
        model = Custom(main_logger, config.NETWORK_PARAMS)
    elif select_net == 'custom_att':
        model = CustomAtt(main_logger, config.NETWORK_PARAMS)
    elif select_net == 'densenet':
        model = models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    # model.state_dict()
    # list(model.parameters())
    log_network_params(model, main_logger)

    return model


def setup(dataset_dir, feature_experiment, data_mode):
    """
    Creates the necessary directories, data folds, logger, and model to be
    used in the experiment. It also determines whether a previous checkpoint
    has been saved.

    Inputs
        dataset_dir: str - The location of the dataset
        feature_experiment: str - The type of features used in this experiment
        data_mode: str - Set to sub if using some of the training data as
                   validation data or set to complete if using the complete
                   training data and no validation
        audio_mode_is_concat_not_shorten: bool - Set False if the data is
                                          shortened to the shortest clip in the
                                          dataset
        make_dataset_equal: bool - Set True if the dataset should be
                            subsampled in order to balance it

    Outputs
        main_logger: logger - The logger to be used to record information
        model: obj - The model to be used for training during the experiment
        checkpoint_run: str - The location of the last saved checkpoint
        checkpoint: bool - True if loading from a saved checkpoint
        next_fold: bool - If loading from a checkpoint is suspected but the
                   current fold experiment has been completed
    """
    reproducibility(chosen_seed)
    checkpoint_run = None
    checkpoint = False
    next_fold = False
    next_exp = False
    if not os.path.exists(features_dir):
        print('There is no folder and therefore no database created. '
              'Create the database first')
        sys.exit()
    if os.path.exists(current_dir) and os.path.exists(model_dir) and debug:
        shutil.rmtree(current_dir, ignore_errors=False, onerror=None)
    # THIS WILL DELETE EVERYTHING IN THE CURRENT WORKSPACE #
    if not os.path.exists(data_fold_dir):
        os.makedirs(data_fold_dir)
        os.makedirs(data_fold_dir_equal)
        dataset_processing.partition_dataset(workspace_main_dir,
                                             feature_experiment,
                                             features_dir,
                                             sub_dir,
                                             current_dir,
                                             data_mode,
                                             dataset_dir,
                                             config_file,
                                             total_folds)

    if os.path.exists(current_dir) and os.path.exists(model_dir):
        temp_dirs = os.listdir(model_dir)
        temp_dirs = natsort.natsorted(temp_dirs, reverse=True)
        temp_dirs = [d for d in temp_dirs if '.pth' in d]
        if len(temp_dirs) == 0:
            pass
        else:
            if int(temp_dirs[0].split('_')[1]) == final_iteration:
                directory = model_dir.split('/')[-1]
                final_directory = model_dir.replace(directory, 'Fold_'+str(total_folds))
                if os.path.exists(final_directory):
                    temp_dirs2 = os.listdir(final_directory)
                    temp_dirs2 = natsort.natsorted(temp_dirs2, reverse=True)
                    temp_dirs2 = [d for d in temp_dirs2 if '.pth' in d]
                    if int(temp_dirs2[0].split('_')[1]) == final_iteration:
                        if i == exp_runthrough-1:
                            print(f"A directory at this location exists: {current_dir}")
                            sys.exit()
                        else:
                            next_exp = True
                            return None, None, None, None, next_fold, next_exp
                    else:
                        next_fold = True
                        return None, None, None, None, next_fold, next_exp
                else:
                    next_fold = True
                    return None, None, None, None, next_fold, next_exp
            else:
                print(f"Current directory exists but experiment not finished")
                print(f"Loading from checkpoint: {int(temp_dirs[0].split('_')[1])}")
                checkpoint_run = os.path.join(model_dir, temp_dirs[0])
                checkpoint = True
    elif not os.path.exists(current_dir):
        os.mkdir(current_dir)
        util.create_directories(current_dir, config.EXP_FOLDERS)
        os.mkdir(model_dir)
    elif os.path.exists(current_dir) and not os.path.exists(model_dir):
        os.mkdir(model_dir)

    main_logger = logging_info()

    model = create_model(main_logger)

    return main_logger, model, checkpoint_run, checkpoint, next_fold, next_exp


def log_network_params(model, main_logger):
    """
    Logs the network architecture for inspection post experiment

    Input
        params: dictionary - The network configuration from the config file
        main_logger: logger - Used to record the network architecture
    """
    main_logger.info('List of Network Parameters')
    conv_count = 1
    fc_count = 1

    for name, param in model.named_parameters():
        if param.requires_grad:
            main_logger.info(f"Name: {name}, \nParam.data: {param.data.shape}")
    if model.named_children():
        for child in model.named_children():
            main_logger.info(child)
    # for p in params:
    #     if len(list(p.size())) > 3:
    #         main_logger.info(f"Conv_{conv_count} : {list(p.size())}")
    #         conv_count += 1
    #     elif 1 < len(list(p.size())) < 3:
    #         main_logger.info(f"FC_{fc_count}: {list(p.size())}")
    #         fc_count += 1


def record_top_results2(current_results, scores, epoch):
    """
    Function to record the best validation F1-Score up to the current epoch.
    More accurate than the alternate function record_top_results

    Inputs:
        current_results: list - current epoch results
        scores: tuple - contains the best results for the experiment
        epoch: int - The current epoch

    Output
        best_res: list - updated best result and epoch of discovery
    """
    if current_results[8] > .86:
        train_f = current_results[9] / 4
        train_loss = current_results[10] / 10
        dev_f = current_results[-6]
        dev_loss = current_results[-5] / 10
        total = train_f - train_loss + dev_f - dev_loss
        if total > scores[0]:
            best_res = [total, current_results[8], current_results[0],
                        current_results[1], current_results[9],
                        current_results[6], current_results[7],
                        current_results[10], current_results[23],
                        current_results[15], current_results[16], dev_f,
                        current_results[21], current_results[22],
                        current_results[25], epoch]
        else:
            best_res = scores
    else:
        best_res = scores

    return best_res


def record_top_results(current_results, scores, epoch):
    """
    Checks the current results against the previous in the experiment and
    updates the respective accuracy metrics the current result is better than
    the best previous result

    Inputs
        current_results: list - Output of the current accuracy metric results
        scores: tuple - Record of the best results to data
        epoch: int - The current epoch

    Output
        best_acc: list - updated best accuracy
        best_fscore: list - updated best F1-Score
        best_loss: list - updated best loss
    """
    if current_results[0] > scores[0][0] and epoch > 7:
        best_acc = [current_results[0], epoch]
    else:
        best_acc = scores[0]
    if current_results[1] > scores[1][0] and epoch > 7:
        best_fscore = [current_results[1], epoch]
    else:
        best_fscore = scores[1]
    if current_results[2] < scores[2][0] and epoch > 7:
        best_loss = [current_results[2], epoch]
    else:
        best_loss = scores[2]

    return best_acc, best_fscore, best_loss


def initialiser(test_value):
    """
    Used to set a bool to True for the initialisation of some function or
    variable

    Input
        test_value: int - If set to 1 then this is the initial condition
                    otherwise, already initialised

    Output
        bool - True if this is the initialisation case
    """
    if test_value == 1:
        return True
    else:
        return False


def compile_train_val_pred(train_res, val_res, comp_train, comp_val, epoch,
                           net_params):
    """
    Used to group the latest results for both the training and the validation
    set into their respective complete results array

    Inputs
         train_res: numpy.array - The current results for this epoch
         val_res: numpy.array - The current results for this epoch
         comp_train: numpy.array - The total recorded results
         comp_val: numpy.array - The total recorded results
         epoch: int - The current epoch used for initialisation
         net_params: dictionary - Holds the model configurations

    Outputs
        comp_train: numpy.array - The updated complete results
        comp_val - numpy.array - The updated complete results
    """
    # 3D matrix ('Num_segments_batches', 'pred+label', 'epochs')
    if epoch == 1:
        comp_train = train_res
        comp_val = val_res
    else:
        if train_res.shape[0] != comp_train.shape[0]:
            difference = comp_train.shape[0] - train_res.shape[0]
            if 'SOFTMAX_1' in net_params:
                train_res = np.vstack((train_res, np.zeros((difference, 3))))
            else:
                train_res = np.vstack((train_res, np.zeros((difference, 2))))

        comp_train = np.dstack((comp_train, train_res))
        comp_val = np.dstack((comp_val, val_res))

    return comp_train, comp_val


def update_complete_results(complete_results, avg_counter,
                            placeholder, best_scores, best_scores2):
    """
    Finalises the complete results dataframe by calculating the mean of the 2
    class scores for accuracy and F1-Score and in the case of the training
    data, divides the results by the number of iterations in order to get the
    average results from the current epoch (previously updated by accumulation)
    Also obtains the best scores for the model.

    Inputs
        complete_results: dataframe - holds the complete results from the
                          experiment so far
        label: str - set to train or dev
        avg_counter: int - used in train mode to average the recorded results
                     for the current epoch
        placeholder: Essentially the number of epochs (but can be used in
                     iteration mode)
        best_scores: tuple - Contains the best scores so far and the
                     respective epochs
        best_scores2: list - More accurate representation of the best score,
                      gives epoch for best validation F1-Score

    Outputs
        complete_results: dataframe - Updated version of the complete results
        best_scores: tuple - Updated version of best_scores
        best_scores2: list - Updated version of best_scores2
    """
    complete_results[0:11] = complete_results[0:11] / avg_counter
    # Accuracy Mean
    complete_results[8] = np.mean(complete_results[0:2])
    complete_results[23] = np.mean(complete_results[15:17])
    # FScore Mean
    complete_results[9] = np.mean(complete_results[6:8])
    complete_results[24] = np.mean(complete_results[21:23])
    print_log_results(placeholder, complete_results[0:15], 'train')
    print_log_results(placeholder, complete_results[15:], 'dev')
    best_scores[0:3] = record_top_results(complete_results[8:11],
                                          best_scores[0:3],
                                          placeholder)
    best_scores[3:] = record_top_results(complete_results[23:26],
                                         best_scores[3:],
                                         placeholder)

    best_scores2 = record_top_results2(complete_results,
                                       best_scores2,
                                       placeholder)

    return complete_results, best_scores, best_scores2


def prediction_and_accuracy(batch_output, batch_labels, initial_condition,
                            num_of_classes, complete_results, loss,
                            per_epoch_pred, config, f_score_average=None):
    """
    Calculates the accuracy (including F1-Score) of the predictions from a
    model. Also the True Negatives, False Negatives, True Positives, and False
    Positives are calculated. These results are stored along with results
    from previous epochs.

    Input
        batch_output: The output from the model
        batch_labels: The respective labels for the batched output
        initial_condition: Bool - True if this is the first instance to set
                           up the variables for logging accuracy
        num_of_classes: The number of classes in this dataset
        complete_results: Dataframe of the complete results obtained
        loss: The value of the loss from the current epoch
        per_epoch_pred: Combined batch outputs and labels for record keeping
        config: The config file for the current experiment
        f_score_average: The type of averaging to be used fro the F1-Score (
                         Macro, Micro, or None

    Output
        complete_results: Dataframe of the complete results to the current epoch
        per_epoch_pred: Combined results of batch outputs and labels for
                        current epoch
    """
    if type(batch_output) is not np.ndarray:
        batch_output = batch_output.data.cpu().numpy()
        batch_labels = batch_labels.data.cpu().numpy()

    if len(batch_output.shape) == 1:
        batch_output = batch_output.reshape(-1, 1)
    if len(batch_labels.shape) == 1:
        batch_labels = batch_labels.reshape(-1, 1)
    if initial_condition:
        per_epoch_pred = np.hstack((batch_output, batch_labels))
    else:
        temp_stack = np.hstack((batch_output, batch_labels))
        per_epoch_pred = np.vstack((per_epoch_pred, temp_stack))

    if 'SIGMOID_1' in config.NETWORK_PARAMS:
        if config.NETWORK_PARAMS['SIGMOID_1'] == 'unnorm':
            prediction = 1 - np.round(batch_output)
        elif config.NETWORK_PARAMS['SIGMOID_1'] == 'round':
            prediction = np.round(batch_output)
        elif config.NETWORK_PARAMS['SIGMOID_1'] == 'threshold':
            height, width = batch_output.shape
            prediction = np.zeros((height, width))
            for pointer, value in enumerate(batch_output):
                if value >= 0.4:
                    prediction[pointer, :] = 1
                else:
                    prediction[pointer, :] = 0
        else:
            print('Error - set "procedure_with_sig" to unnorm or round')
            sys.exit()
        prediction = prediction.reshape(-1)
    else:
        prediction = np.argmax(batch_output, axis=1)

    if len(batch_labels.shape) > 1:
        batch_labels = batch_labels.reshape(-1)
    if batch_labels.dtype == 'float32':
        batch_labels = batch_labels.astype(np.long)

    acc, fscore, tn_fp_fn_tp = calculate_accuracy(batch_labels, prediction,
                                                  num_of_classes,
                                                  f_score_average)
    complete_results[0:2] += acc
    complete_results[2:8] += np.array(fscore[0:3]).reshape(1, -1)[0]
    complete_results[10] += loss
    complete_results[11:15] += tn_fp_fn_tp

    return complete_results, per_epoch_pred


def print_log_results(epoch, results, data_type):
    """
    Used to print/log results after every epoch

    Inputs
        epoch: int - The current epoch
        results: numpy.array - The current results
        data_type: str - Set to train, val, or test
    """
    print('\n', config_dataset.SEPARATOR)
    print(f"{data_type} accuracy at epoch: {epoch}\n{data_type} Accuracy: Mean:"
          f" {np.round(results[8], 3)} - {np.round(results[0:2], 3)}, "
          f"F1_Score: Mean: {np.round(results[9], 3)} -"
          f" {np.round(results[6:8], 3)}, Loss: {np.round(results[10], 3)}")

    print(config_dataset.SEPARATOR, '\n')

    main_logger.info(f"\n{config_dataset.SEPARATOR}{config_dataset.SEPARATOR}")
    main_logger.info(f"{data_type} accuracy at epoch: {epoch}\n{data_type} "
                     f"Accuracy: Mean: {np.round(results[8], 3)} -"
                     f" {np.round(results[0:2], 3)}, F1_Score: Mean:"
                     f" {np.round(results[9], 3)},"
                     f" {np.round(results[6:8], 3)}, Loss:"
                     f" {np.round(results[10], 3)}")
    main_logger.info(f"{config_dataset.SEPARATOR}{config_dataset.SEPARATOR}\n")


def final_organisation(scores, train_pred, val_pred, df, patience,
                       epoch, scores2, workspace_files_dir):
    """
    Records final information with the logger such as the best scores for
    training and validation and saves/copies files from the current
    experiment into the saved model directory for future analysis. The
    complete results to the current epoch are saved for checkpoints or future
    analysis

    Inputs
        scores: list - The best scores from the training and validation results
        train_pred: numpy.array - Record of the complete outputs of the
                    network for every epoch
        val_pred: numpy.array - Record of the complete outputs of the
                  network for every epoch
        df: pandas.dataframe - The complete results for every epoch
        patience: int - Used to record if early stopping was implemented
        epoch: int - The current epoch
        scores2: list - More accurate version of scores. Only holds the
                 best validation F1-Score location
        workspace_files_dir: str - Location of the programme code
    """
    main_logger.info(f"Best Train Acc: {scores[0]}\nBest Train Fscore:"
                     f" {scores[1]}\nBest Train Loss: {scores[2]}\nBest Val "
                     f"Acc: {scores[3]}\nBest Val Fscore: {scores[4]}\nBest "
                     f"Val Loss: {scores[5]}")
    main_logger.info(f"\nBEST SCORES2 at Epoch: {scores2[-1]}\nTrain Acc2:"
                     f" {scores2[1]}\nTrain Fscore2: {scores2[4]}\nTrain "
                     f"Loss2: {scores2[7]}\nVal Acc2: {scores2[8]}\nVal "
                     f"Fscore2: {scores2[11]}\nVal Loss2: {scores2[14]}")
    main_logger.info(f"\nscores: {scores2[1:-1]}")

    if epoch == final_iteration:
        main_logger.info(f"System will exit as the total number of "
                         f"epochs has been reached  {final_iteration}")
    else:
        main_logger.info(f"System will exit as the validation loss "
                         f"has not improved for {patience} epochs")
    print(f"System will exit as the validation loss has not "
          "improved for {patience} epochs")
    utilities.save_model_outputs(model_dir, df, train_pred, val_pred, scores,
                                 scores2[1:])

    copyfile(workspace_files_dir + '/models_pytorch.py',
             current_dir + '/model_architecture.py')
    copyfile(workspace_files_dir + '/main.py', current_dir + '/main.py')
    copyfile(workspace_files_dir + '/data_loader/data_gen.py',
             current_dir + '/data_gen.py')
    copyfile(workspace_files_dir + '/data_loader/organiser.py',
             current_dir + '/organiser.py')
    copyfile(workspace_files_dir + '/' + config_file + '.py', current_dir +
             '/' + config_file + '.py')


def reduce_learning_rate(optimizer):
    """
    Reduce the learning rate of the optimiser for training

    Input
        optimiser: obj - The optimiser setup at the start of the experiment
    """
    learning_rate_reducer = 0.9
    for param_group in optimizer.param_groups:
        print('Reducing Learning rate from: ', param_group['lr'],
              ' to ', param_group['lr'] * learning_rate_reducer)
        main_logger.info(f"Reducing Learning rate from: "
                         f"{param_group['lr']}, to "
                         f"{param_group['lr'] * learning_rate_reducer}")
        param_group['lr'] *= learning_rate_reducer


def reproducibility(chosen_seed):
    """
    The is required for reproducible experimentation. It sets the random
    generators for the different libraries used to a specific, user chosen
    seed.

    Input
        chosen_seed: int - The seed chosen for this experiment
    """
    torch.manual_seed(chosen_seed)
    torch.cuda.manual_seed_all(chosen_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(chosen_seed)
    random.seed(chosen_seed)


def get_rnn_feats(net_params, current_batch_size):
    """
    Function to setup the initial conditions to be used in any RNN layers.
    h0 will need to be created and if LSTM is used then c0 will also be
    created. The layers are increased if the RNN is bidirectional

    Inputs
        net_params: Used to determine what type of RNN is used
        current_batch_size: The size of the current batch being processed

    Output
        hc: The initialised hidden layers for the RNN
    """
    if 'LSTM_1' in net_params or 'GRU_1' in net_params:
        if 'LSTM_1' in net_params:
            rnn_feats = net_params['LSTM_1'][1]
            bidi = net_params['LSTM_1'][-1]
            layers = net_params['LSTM_1'][2]
        else:
            rnn_feats = net_params['GRU_1'][1]
            bidi = net_params['GRU_1'][-1]
            layers = net_params['GRU_1'][2]
        if bidi:
            b = 2 * layers
        else:
            b = 1 * layers
        h0 = np.random.randn(b, current_batch_size, rnn_feats)
        h0 = mu.create_tensor_data(h0, cuda, select_net)
        if 'LSTM_1' in net_params:
            c0 = np.random.randn(b, current_batch_size, rnn_feats)
            c0 = mu.create_tensor_data(c0, cuda, select_net)
            hc = (h0, c0)
        else:
            hc = h0
    else:
        hc = (torch.zeros(1), torch.zeros(1))

    return hc


def collate_net_outputs(output, output_att=None, net_params='SOFTMAX_1',
                        learning_procedure='soft_mv',
                        avg_setting='arithmetic', current_batch_size=20,
                        iterator=0, num=1):
    """
    Processes the output of the model depending on user specified options
    such as whether hard or soft majority vote is used, and whether geometric
    or arithmetic averaging are to be used. Also handles the situation where
    an attention mechanism is used.

    Input
        output: Raw output from the model of this experiment
        output_att: Raw output of the attention mechanism for this experiment
        net_params: Dictionary containing the model configurations
        learning_procedure: Soft or hard majority vote
        avg_setting: Geometric or arithmetic averaging setting
        current_batch_size: The current size of the batched output

    Output
        output: Processed output
    """
    net_last_layer = [k for k in net_params.keys()][-1]
    if 'ATTENTION_global' in net_params or 'ATTENTION_1' in net_params:
        if iterator+1 == num:
            if 'ATTENTION_global' in net_params:
                soft = torch.nn.Softmax(dim=-1)
                output_att = soft(output_att)
                attended = output * output_att
                output = torch.sum(attended,
                                   dim=1).reshape(current_batch_size,
                                                  -1)
            output = torch.clamp(output, min=0, max=1)
        else:
            if cuda:
                output = output.cpu()
            return output
    elif learning_procedure == 'hard_mv':
        if avg_setting == 'geometric':
            if net_last_layer == 'SIGMOID_1':
                output = torch.cat((output, 1 - output),
                                   dim=1)
            output = torch.clamp(torch.round(output)+EPS,
                                 min=0, max=1)
            output = torch.log(output)
        else:
            output = torch.round(output)
    else:
        if avg_setting == 'geometric':
            if net_last_layer == 'SIGMOID_1':
                output = torch.cat((output, 1 - output),
                                   dim=1)
            output = torch.log(output)
        else:
            output = output

    if cuda:
        output = output.cpu()

    return output


def get_output_from_model(data, net_type, net_params, learning_procedure,
                          learning_procedure_decider, locator=[],
                          label='', recurrent_out='whole',
                          convert_to_image=False):
    """
    Pushes the batched data to the user specified neural network and
    depending on the settings the output will be processed regarding
    how long the files are in the batch. The high level options are
    random_sample, chunked_file, and whole_file. The output processing
    depends also on whether soft majority vote or hard majority vote are
    selected along with the type of averaging (arithmetic and geometric) and
    the final layer of the network (softmax or sigmoid)

    Inputs:
        data: Data to be pushed to the model
        net_type: What model is used?
        net_params: The model configuration - includes layers and filter data
        learning_procedure: How is the data processed: random_sample,
                            chunked_file, or whole_file
        learning_procedure_decider: Soft majority vote or hard majority vote.
                                    If single files are used instead of sequence
                                    data, soft majority vote is used
        locator: Array of the lengths of the files in the batch
        label: Set to train or dev depending on the section of experiment
        recurrent_out: If RNN is used, how is the output processed?

    Output
        output: The output of the model from the input batch data
    """
    net_last_layer = [k for k in net_params.keys()][-1]
    if net_type == 'densenet':
        output = model(data)
        lsm = torch.nn.LogSoftmax(dim=-1)
        output = lsm(output)
    elif net_type == 'custom' or net_type == 'custom_att':
        if learning_procedure == 'random_sample':
            segments = 1
        else:
            segments = np.max(locator)

        current_batch_size = data.shape[0] // segments
        if 'SOFTMAX_1' in net_params:
            output = torch.zeros(current_batch_size, 2)
        else:
            output = torch.zeros(current_batch_size, 1)

        placeholder = 0
        hc = get_rnn_feats(net_params, current_batch_size)

        output_net = torch.zeros(current_batch_size, segments)
        attout_net = torch.zeros(current_batch_size, segments)
        for p in range(segments):
            current_data = data[placeholder:placeholder+current_batch_size]
            current_data = mu.create_tensor_data(current_data,
                                                 cuda,
                                                 select_net)
            if 'ATTENTION_global' in net_params:
                temp_out, hc, temp_att = model(current_data, net_params,
                                               convert_to_image, hc,
                                               recurrent_out, label)
                if label == 'dev':
                    zero_out_index = torch.ones((temp_out.shape[0], 1))
                    for pos, loc in enumerate(locator):
                        if p >= loc:
                            zero_out_index[pos] = 0
                    zero_out_index = mu.create_tensor_data(
                        zero_out_index, cuda, net_type)

                    temp_out = temp_out * zero_out_index
                    temp_att = temp_att * zero_out_index

                output_net[:, p] = temp_out.reshape(current_batch_size)
                attout_net[:, p] = temp_att.reshape(current_batch_size)
            else:
                if 'LSTM_1' in net_params or 'GRU_1' in net_params:
                    output_net, hc, _ = model(current_data, net_params,
                                              convert_to_image, hc,
                                              recurrent_out, label, locator)
                else:
                    output_net, _, _ = model(current_data, net_params,
                                             convert_to_image,
                                             recurrent_out=recurrent_out,
                                             label=label, locator=locator)
                if label == 'dev':
                    zero_out_index = torch.ones((output_net.shape[0], 1))
                    for pos, loc in enumerate(locator):
                        if p >= loc:
                            zero_out_index[pos] = 0
                    zero_out_index = mu.create_tensor_data(
                        zero_out_index, cuda, net_type)

                    output_net = output_net * zero_out_index
                attout_net = None

            temp_out = collate_net_outputs(output_net, attout_net,
                                           net_params,
                                           learning_procedure_decider,
                                           averaging,
                                           current_batch_size, p, segments)
            if 'ATTENTION_global' in net_params and p+1 != segments:
                pass
            else:
                output = output + temp_out

            # output = pre_output + output
            placeholder += current_batch_size

        if 'ATTENTION_global' in net_params:
            pass
        elif 'ATTENTION_1' in net_params:
            if label == 'dev':
                if output.dim() > 1:
                    locator = torch.Tensor(locator).view(-1, 1)
                else:
                    locator = torch.Tensor(locator)
                output = output / locator
            else:
                output = output / segments
            # output = torch.clamp(output, min=0, max=1)
        else:
            if learning_procedure == 'whole_file' and label == 'dev':
                if output.dim() > 1:
                    locator = torch.Tensor(locator).view(-1, 1)
                else:
                    locator = torch.Tensor(locator)
                output = output / locator
            else:
                output = output / segments
                if learning_procedure_decider == 'hard_mv' and averaging == \
                        'arithmetic' and net_last_layer == 'SIGMOID_1':
                    output = torch.clamp(output+EPS, min=0, max=1)

    if 'SOFTMAX_1' in net_params:
        output = torch.log(output)

    return output


def tensorboard_visual(tb_writer, df, epoch_iter):
    """
    Outputs current experiment results to Tensorboard to be visualised

    Inputs:
        tb_writer: obj - Tensorboard object which collects the information
        df: pandas.dataframe - Holds complete results of experiment
        epoch_iter: int - The current epoch of the experiment
    """
    col = ['train_mean_acc', 'train_mean_fscore', 'train_loss',
           'val_mean_acc', 'val_mean_fscore', 'val_loss']
    t_acc = df[col[0]].tolist()[-1]
    t_f_score = df[col[1]].tolist()[-1]
    t_loss = df[col[2]].tolist()[-1]
    v_acc = df[col[3]].tolist()[-1]
    v_f_score = df[col[4]].tolist()[-1]
    v_loss = df[col[5]].tolist()[-1]

    tb_writer.add_scalar('Train_Acc', np.array(t_acc), epoch_iter)
    tb_writer.add_scalar('Val_Acc', np.array(v_acc), epoch_iter)
    tb_writer.add_scalar('Train_F-Score', np.array(t_f_score), epoch_iter)
    tb_writer.add_scalar('Val_F-Score', np.array(v_f_score), epoch_iter)
    tb_writer.add_scalar('Train_Loss', np.array(t_loss), epoch_iter)
    tb_writer.add_scalar('Val_Loss', np.array(v_loss), epoch_iter)

    f = plot_graph(epoch_iter, df, final_iteration, model_dir, vis=vis)

    tb_writer.add_figure('predictions vs. actuals', f, epoch_iter)
    tb_writer.file_writer.flush()


def begin_evaluation(mode, epoch, reset, iteration, iteration_epoch):
    """
    Determines whether to start the validation processing. This is determined
    by the a change in epoch and reset

    Inputs
        mode: str - Is the mode in epoch selection or iteration selection
        epoch: int - The current epoch of the experiment
        reset: Bool - True if the end of a training phase has occurred
        iteration: int - The current iteration of the experiment
        iteration_epoch: int - The number of iterations equivalent to the
                         number of epochs if it were in epoch mode

    Output
        Bool: True if validation set should be processed
    """
    if mode == 'epoch':
        if epoch > 0 and reset:
            return True
    elif mode == 'iteration':
        it = iteration + 1
        if it % iteration_epoch == 0 and iteration > 0:
            return True
    elif mode is None:
        print('Wrong Mode Selected. Choose either epoch or iteration in the '
              'config file.')
        print('The program will now exit')
        sys.exit()


def train(model, feature_experiment, workspace_files_dir, convert_to_im):
    """
    Sets up the experiment and runs the experiment. The training batches are
    loaded and pushed to the model, the resuls are analysed until the next
    epoch. At this point, the validation set is run in the same manner.
    Results and outputs are collated and recorded for future analysis and
    checkpointing

    Input
        model: obj - The model to be used for experimentation
        feature_experiment: str - The type of features for this experiment
        workspace_files_dir: str - Location of the programme code
        convert_to_im: book - Set True if the data is to be converted to 3D
    """
    num_of_classes = len(config_dataset.LABELS)

    writers = SummaryWriter(f"./runs/{feature_experiment}_{sub_dir}_F{current_fold}")
    generator, cw_train, cw_dev, total_num_zeros, total_num_ones, \
        = organiser.run(config, main_logger, current_fold, checkpoint)

    print('Generating data for training')
    learning_rate = config.EXPERIMENT_DETAILS['LEARNING_RATE']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    main_logger.info(f"Optimiser: ADAM. Learning Rate: {learning_rate}")

    if checkpoint:
        start_epoch, data_saver = utilities.load_model(checkpoint_run, model,
                                                       optimizer)
        df, comp_train_pred, comp_val_pred, best_scores, best_scores2 = \
            utilities.load_model_outputs(model_dir)
        counter = start_epoch
    else:
        start_epoch = 0
        # train_acc, train_fscore, train_loss, val_acc, val_fscore, val_loss
        best_scores = [[0, 0], [0, 0], [1e7, 0], [0, 0], [0, 0], [1e7, 0]]
        best_scores2 = [0] * 16
        comp_train_pred = comp_val_pred = 0
        df = pd.DataFrame(columns=config_dataset.COLUMN_NAMES)
        data_saver = {}

    avg_counter = per_epoch_train_pred = 0
    # Train/Val, Accuracy, Precision, Recall, Fscore, Loss(single), mean_acc/f
    complete_results = np.zeros(30)

    net_params = config.NETWORK_PARAMS
    recurrent_out = config.RECURRENT_OUT
    learn_rate_factor = 6
    print('Beginning Training')
    for (iteration, (batch_data, batch_labels, epoch, reset, train_batch_loc,
                     data_saver)) in enumerate(generator.generate_train_data(
        start_epoch, data_saver)):
        start_timer = time.time()
        avg_counter += 1

        model.train()
        batch_labels = mu.create_tensor_data(batch_labels,
                                             cuda,
                                             select_net,
                                             True)
        batch_output = get_output_from_model(data=batch_data,
                                             net_type=select_net,
                                             net_params=net_params,
                                             learning_procedure=learning_procedure_train,
                                             learning_procedure_decider=learning_procedure_decider_train,
                                             locator=train_batch_loc,
                                             label='train',
                                             recurrent_out=recurrent_out,
                                             convert_to_image=convert_to_im)

        train_loss = mu.calculate_loss(batch_output, batch_labels, averaging,
                                       cw_train, net_params)

        # Zero the gradient and Backprop the loss through the network
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        init = initialiser(avg_counter)
        complete_results[0:15], per_epoch_train_pred = \
            prediction_and_accuracy(batch_output, batch_labels, init,
                                    num_of_classes, complete_results[0:15],
                                    train_loss, per_epoch_train_pred, config)

        begin_evaluate = begin_evaluation(analysis_mode, epoch, reset,
                                          iteration, iteration_epoch)
        if begin_evaluate:
            if analysis_mode == 'epoch':
                print(best_scores2)
                placeholder = epoch
            else:
                counter += 1
                placeholder = counter

            if validate:
                print(f"Evaluating - Training at epoch: {placeholder}, "
                      f"iteration: {iteration}\nTime taken for training: ' "
                      f"{time.time()-start_timer}")
                main_logger.info(f"Time taken for training:"
                                 f" {time.time() - start_timer} at iteration:"
                                 f" {iteration}")
                print('Evaluating - Development at epoch: ', placeholder)
                (complete_results[15:], per_epoch_val_pred) = evaluate(
                    generator=generator,
                    data_type='dev',
                    class_weights=cw_dev,
                    comp_res=complete_results[15:],
                    class_num=num_of_classes,
                    f_score_average=None,
                    averaging=averaging,
                    net_p=net_params,
                    recurrent_out=recurrent_out,
                    epochs=start_epoch,
                    convert_im=convert_to_im)

                complete_results, best_scores, best_scores2 = \
                    update_complete_results(complete_results, avg_counter,
                                            placeholder, best_scores,
                                            best_scores2)
                avg_counter = 0

                df.loc[placeholder-1] = complete_results

                complete_results = np.zeros(30)
                plot_graph(placeholder, df, final_iteration, model_dir,
                           vis=vis)

            comp_train_pred, comp_val_pred = compile_train_val_pred(
                per_epoch_train_pred,
                per_epoch_val_pred,
                comp_train_pred,
                comp_val_pred,
                placeholder,
                net_params)

            # Save model
            utilities.save_model(placeholder, model, optimizer, main_logger,
                                 model_dir, data_saver)
            tensorboard_visual(writers, df, epoch)
            utilities.save_model_outputs(model_dir, df, comp_train_pred,
                                         comp_val_pred, best_scores,
                                         best_scores2)
            # Reduce learning rate
            if placeholder % learn_rate_factor == 0:
                reduce_learning_rate(optimizer)

            # Stop learning
            patience = final_iteration+1
            if placeholder % patience == 0:
                writers.close()
                reference = df['val_loss'].tolist()[-patience]
                trial = df['val_loss'].tolist()[-1]
                if not trial < reference:
                    print(f"Val_loss - Patience {reference}\nVal loss - "
                          f"Current {trial}")
                    final_organisation(best_scores, comp_train_pred,
                                       comp_val_pred, df, patience,
                                       placeholder, best_scores2,
                                       workspace_files_dir)

                    plot_graph(placeholder, df, final_iteration, model_dir,
                               early_stopper=True, vis=vis)
                    break
            elif placeholder == final_iteration:
                final_organisation(best_scores, comp_train_pred,
                                   comp_val_pred, df, patience, placeholder,
                                   best_scores2, workspace_files_dir)
                break


def test():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='mode')

    parser_train = sub_parser.add_parser('train')
    parser_train.add_argument('--validate', action='store_true', default=False,
                              help='set whether we want to use a validation set')
    parser_train.add_argument('--vis', action='store_true', default=False,
                              help='determine whether model graph is output')
    parser_train.add_argument('--cuda', action='store_true', default=False,
                              help='pass --cuda if you want to run on GPU')
    parser_train.add_argument('--debug', action='store_true', default=False,
                              help='set the program to run in debug mode '
                                   'means, that the most recent folder will '
                                   'be deleted automatically to speed up '
                                   'debugging')
    parser_train.add_argument('--limited_memory', action='store_true',
                              default=False, help='Set to true if working '
                                                  'with less than 32GB RAM')

    parser_test = sub_parser.add_parser('test')
    parser_test.add_argument('--vis', action='store_true', default=False,
                             help='determine whether model graph is output')
    parser_test.add_argument('--cuda', action='store_true', default=False,
                             help='pass --cuda if you want to run on GPU')

    args = parser.parse_args()

    debug = args.debug
    validate = args.validate
    vis = args.vis
    cuda = args.cuda
    cf = 'config'

    config_extensions = ['1']
    for ext in config_extensions:
        config_file = cf+'_'+str(ext)
        if not os.path.exists(config_file):
            config = importlib.import_module(config_file)
        else:
            config = importlib.import_module(cf)

        workspace_main_dir = config.WORKSPACE_MAIN_DIR
        features_dir = os.path.join(workspace_main_dir, config.FOLDER_NAME)
        print('feature_dir:', features_dir)
        total_folds = config.EXPERIMENT_DETAILS['TOTAL_FOLDS']
        chosen_seed = config.EXPERIMENT_DETAILS['SEED']
        exp_name = config.EXP_NAME
        analysis_mode = config.ANALYSIS_MODE
        learning_procedure_decider_train = config.LEARNING_PROCEDURE_DECIDER_TRAIN
        learning_procedure_decider_dev = config.LEARNING_PROCEDURE_DECIDER_DEV
        learning_procedure_train = config.LEARNING_PROCEDURE_TRAIN
        learning_procedure_dev = config.LEARNING_PROCEDURE_DEV
        averaging = config.AVERAGING
        select_net = config.EXPERIMENT_DETAILS['NETWORK']

        data_fold_dir = os.path.join(workspace_main_dir,
                                     'data_folds_'+str(total_folds))
        data_fold_dir_equal = data_fold_dir + '_equal'

        iteration_epoch = config.EXPERIMENT_DETAILS['ITERATION_EPOCH']
        if analysis_mode == 'epoch':
            final_iteration = config.EXPERIMENT_DETAILS['TOTAL_EPOCHS']
        elif analysis_mode == 'iteration':
            final_iteration = config.EXPERIMENT_DETAILS['TOTAL_ITERATIONS']
            final_iteration = round(final_iteration / iteration_epoch)

        folder_extensions = ['a', 'b', 'c', 'd', 'e']
        exp_runthrough = 5
        if args.mode == 'train':
            for i in range(exp_runthrough):
                sub_dir = config.EXPERIMENT_DETAILS['SUB_DIR']
                sub_dir = sub_dir+folder_extensions[i]
                for current_fold in range(1, total_folds+1):
                    current_dir = os.path.join(features_dir,
                                               sub_dir + '_' + exp_name)
                    model_dir = os.path.join(current_dir,
                                             'model',
                                             f"Fold_{current_fold}")
                    main_logger, model, checkpoint_run, checkpoint, \
                        next_fold, next_exp = setup(config.DATASET,
                                              config.EXPERIMENT_DETAILS[
                                               'FEATURE_EXP'],
                                              config.EXPERIMENT_DETAILS['MODE'])
                    if next_exp:
                        break

                    if next_fold:
                        continue

                    comp_start_time = time.time()
                    train(model, config.EXPERIMENT_DETAILS['FEATURE_EXP'],
                          config.WORKSPACE_FILES_DIR,
                          config.EXPERIMENT_DETAILS['CONVERT_TO_IMAGE'])
                    comp_end_time = time.time()
                    complete_time = comp_end_time - comp_start_time
                    main_logger.info(f"Complete time to run model:"
                                     f" {complete_time}")
                    handlers = main_logger.handlers[:]
                    for handler in handlers:
                        handler.close()
                        main_logger.removeHandler(handler)
                chosen_seed += 100
        elif args.mode == 'test':
            test()
        else:
            raise Exception('There has been an error in the input arguments')
