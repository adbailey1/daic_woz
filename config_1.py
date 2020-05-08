import os
import sys


# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw',
                      'AUDIO_MODE_IS_CONCAT_NOT_SHORTEN': True,
                      'MAKE_DATASET_EQUAL': False,
                      'CLASS_WEIGHTS': True,
                      'FEATURE_DIMENSIONS': 55000,
                      'FREQ_BINS': 1,
                      'BATCH_SIZE': 18,
                      'LEARNING_RATE': 1e-3,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 150,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'exp_2',
                      'TOTAL_FOLDS': 4,
                      'MODE': 'sub',
                      'DATASET_IS_BACKGROUND': False,
                      'CONVERT_TO_IMAGE': False,
                      'REMOVE_BACKGROUND': True,
                      'SECONDS_TO_SEGMENT': 30,
                      'WHOLE_TRAIN': 'mean',
                      'NETWORK': 'custom_att'}
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'
# (Channels), Kernel, Stride, Pad, (wn/bn for conv or FC layers)
# GRU - input_size, hidden_size, bidirectional
# (as int), layers, bidirectional (as Bool)
# Attention mechanism always goes first if used!
# 'ATTENTION_1': [type, filter, stride, pad, norm, axis],
# 'ATTENTION_1': ['conv', (1, 128), (1, 1), (0, 0), 'bn', 'time']
NETWORK_PARAMS = {'CONV_1': [[1, 128], (1024), (512), (404), 'bn'],
                  'CONV_2': [[128, 128], (3), (1), (1), 'bn'],
                  'CONV_3': [[128, 128], (3), (1), (1), 'bn'],
                  'POOL_1': [(3), (3), (0)],
                  'DROP_1': .5,
                  'LSTM_1': [128, 128, 3, False],
                  'ATTENTION_1': ['fc', 128, 1, 'time'],
                  'ATTENTION_global': ['fc', 128, 1, 'time'],
                  'HIDDEN_1': [128, 1],
                  'SIGMOID_1': 'threshold'}

# NETWORK_PARAMS = {'CONV_1': [[1, 128], (625), (326), (159), 'bn'],
#                   'CONV_2': [[128, 128], (3), (1), (1), 'bn'],
#                   'POOL_1': [(3), (3), (1)],
#                   'LSTM_1': [128, 128, 1, False],
#                   'DROP_2': 0.5,
#                   'ATTENTION_1': ['fc', 128, 1, 'bn', 'time'],
#                   'HIDDEN_1': [128, 1, 'bn'],
#                   'SIGMOID_1': 'round'}

# FOR USE WITH LSTM/GRU ONLY
# Options: 'whole', 'forward', 'forward_only', 'forward_backward'
RECURRENT_OUT = 'whole'
# whole_file (every file is processed as a whole),
# chunked_file (eg 30s chunks), or random_sample
LEARNING_PROCEDURE_TRAIN = 'chunked_file'
LEARNING_PROCEDURE_DEV = 'chunked_file'
# soft_mv (also used for generic averaging), hard_mv or max_pool
LEARNING_PROCEDURE_DECIDER_TRAIN = 'soft_mv'
LEARNING_PROCEDURE_DECIDER_DEV = 'soft_mv'
# Take logs before (geometric) or after (arithmetic) averaging
AVERAGING = 'arithmetic'

# These values should be the same as those used to create the database
# If raw audio is used, you might want to set these to the conv kernel and
# stride values
WINDOW_SIZE = 1024
HOP_SIZE = 512
OVERLAP = int((HOP_SIZE / WINDOW_SIZE) * 100)

FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']:
    extension = 'concat'
else:
    extension = 'shorten'
if EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']:
    data_eq = '_equalSet'
else:
    data_eq = ''
if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
    bkgnd = '_bkgnd'
else:
    bkgnd = ''

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_WIN_" \
                      f"{str(WINDOW_SIZE)}_OVERLAP_{str(OVERLAP)}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_" \
                      f"{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_with_backgnd_exp"
elif EXPERIMENT_DETAILS['FEATURE_EXP'] == 'raw':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_with_backgnd_exp"
else:
    FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
EXP_NAME = f"{extension}{data_eq}{bkgnd}"

if sys.platform == 'win32':
    DATASET = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif sys.platform == 'linux' and not os.uname()[1] == 'andrew-ubuntu':
    DATASET = os.path.join('/vol/vssp/datasets/singlevideo01/DAIC-WOZ')
    # set the path of the workspace (where the code is)
    WORKSPACE_FILES_DIR = \
        '/user/HS227/ab01814/pycharm_projects/daic_woz_att_raw'
    # set the path of the workspace (where the models/output will be stored)
    WORKSPACE_MAIN_DIR = os.path.join('/vol/research/ab01814_res', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif os.uname()[1] == 'andrew-ubuntu':
    DATASET = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets/DAIC-WOZ'
    WORKSPACE_MAIN_DIR = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2'
    WORKSPACE_FILES_DIR = os.path.join('/home', 'andrew', 'PycharmProjects',
                                       'daic_woz_att_raw')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
else:
    DATASET = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('Users', 'andrewbailey', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'test_split_Depression_AVEC2017.csv')
