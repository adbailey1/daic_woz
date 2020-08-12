import os
import sys


# Set to complete to use all the data
# Set to sub to use training/dev sets only
# FEATURE_EXP: logmel, mel, raw, MFCC, MFCC_concat, or text
# 'NETWORK': 'custom' or 'custom_att' for attention
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'logmel',
                      'AUDIO_MODE_IS_CONCAT_NOT_SHORTEN': True,
                      'MAKE_DATASET_EQUAL': False,
                      'CLASS_WEIGHTS': True,
                      'USE_GENDER_WEIGHTS': False,
                      'FEATURE_DIMENSIONS': 100,
                      'FREQ_BINS': 40,
                      'BATCH_SIZE': 18,
                      'LEARNING_RATE': 1e-3,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'exp_1',
                      'TOTAL_FOLDS': 4,
                      'MODE': 'sub',
                      'DATASET_IS_BACKGROUND': False,
                      'CONVERT_TO_IMAGE': False,
                      'REMOVE_BACKGROUND': True,
                      'SECONDS_TO_SEGMENT': 30,
                      'WHOLE_TRAIN': 'False',
                      'NETWORK': 'custom'}
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'
# Determine the number of times to re-reun the experiment, which corresponds
# to the folder extension for each experiment
FOLDER_EXTENSIONS = ['a', 'b', 'c', 'd', 'e']
EXP_RUNTHROUGH = 5
# Set to 'm' or 'f' to split into male or female respectively
# Otherwise set to '-' to keep both genders in the database
GENDER = '-'

# Channels, Kernel, Stride, Pad
# GRU - input_size - number of channels from last layer (as int), hidden_size
# (as int), layers, bidirectional (as Bool)
# Attention mechanism always goes first if used!
# 'ATTENTION_1': [type, filter, stride, pad, norm, axis],
# 'ATTENTION_1': ['conv' or 'fc', (1, 128), (1, 1), (0, 0), 'bn', 'time' or
# 'freq']
# If using sigmoid set to: unnorm or round or threshold
NETWORK_PARAMS = {'CONV_1': [[1, 128], (40, 3), (1, 1), (0, 1), 'bn'],
                  'POOL_1': [(1, 3), (1, 3), (0, 1)],
                  'DROP_1': 0.5,
                  'LSTM_1': [128, 256, 1, False],  # in, hidden, layers, bidi
                  'HIDDEN_1': [256, 1, 'bn'],
                  'SIGMOID_1': 'round'}

# FOR USE WITH LSTM/GRU ONLY
# Options: 'whole', 'forward', 'forward_only', 'forward_backward'
RECURRENT_OUT = 'forward_only'
# whole_file (every file is processed as a whole),
# chunked_file (eg 30s chunks), or random_sample
LEARNING_PROCEDURE_TRAIN = 'chunked_file'
LEARNING_PROCEDURE_DEV = 'chunked_file'
# soft_mv (also used for generic averaging), hard_mv or max_pool
LEARNING_PROCEDURE_DECIDER_TRAIN = 'soft_mv'
LEARNING_PROCEDURE_DECIDER_DEV = 'soft_mv'
# Take logs before (geometric) or after (arithmetic) averaging
# Only use with output=Softmax
AVERAGING = 'arithmetic'

WINDOW_SIZE = 1024
HOP_SIZE = 512
OVERLAP = int((HOP_SIZE / WINDOW_SIZE) * 100)

EXP_FOLDERS = ['log', 'model', 'condor_logs']

DATASET = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets/DAIC-WOZ'
WORKSPACE_MAIN_DIR = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2'
WORKSPACE_FILES_DIR = os.path.join('/home', 'andrew', 'PycharmProjects',
                                   'daic_woz_att_raw')
TRAIN_SPLIT_PATH = os.path.join(DATASET, 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
