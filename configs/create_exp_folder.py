from configs import config as c

FEATURE_FOLDERS = ['audio_data', c.EXPERIMENT_DETAILS['FEATURE_EXP']]
if c.EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if c.EXPERIMENT_DETAILS['AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']:
    extension = 'concat'
else:
    extension = 'shorten'
if c.EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']:
    data_eq = '_equalSet'
else:
    data_eq = ''
if c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
    bkgnd = '_bkgnd'
else:
    bkgnd = ''

if c.EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or c.EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'mel' or c.EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or c.EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{c.EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(c.EXPERIMENT_DETAILS['FREQ_BINS'])}_exp"
    elif not c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            c.EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{c.EXPERIMENT_DETAILS['FEATURE_EXP']}_{str(c.EXPERIMENT_DETAILS['FREQ_BINS'])}_WIN_" \
                      f"{str(c.WINDOW_SIZE)}_OVERLAP_{str(c.OVERLAP)}_exp"
    elif not c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            c.EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{c.EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_" \
                      f"{str(c.EXPERIMENT_DETAILS['FREQ_BINS'])}_with_backgnd_exp"
elif c.EXPERIMENT_DETAILS['FEATURE_EXP'] == 'raw':
    if c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{c.EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            c.EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{c.EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not c.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            c.EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{c.EXPERIMENT_DETAILS['FEATURE_EXP']}_with_backgnd_exp"
else:
    FOLDER_NAME = f"{c.EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
EXP_NAME = f"{extension}{data_eq}{bkgnd}"
