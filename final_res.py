import pickle
import numpy as np
from pathlib import Path
import os
import csv
import argparse
import pdb


def average_results(starting_epoch, num_folds):
    """
    Loads the different fold results and calculates the average results at a
    specific epoch

    Input
        starting_epoch: int - The epoch to obtain results from
        num_folds: int - How many folds are there in the current experiment
    """
    for i in range(num_folds):
        origin_path = str(Path().absolute())
        path = os.path.join(origin_path, 'model', 'Fold_'+str(i+1),
                            'complete_results.pickle')
        with open(path, 'rb') as f:
            res = pickle.load(f)

        headers = res.keys().to_list()
        res = res.to_numpy()

        positions = [8, 0, 1, 9, 6, 7, 10, 23, 15, 16, 24, 21, 22, 25]
        final_headers = [headers[i] for i in positions]
        res = res[:, positions]
        res = res[starting_epoch:, :]
        res = np.mean(res, axis=0)
        if i == 0:
            combined_res = res
        else:
            combined_res = np.vstack((combined_res, res))
    
    save_file = os.path.join(origin_path, 'cross_val_out.csv')
    with open(save_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(final_headers)
        for i in combined_res:
            writer.writerow(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=20,
                        help='set the epoch of convergence')
    parser.add_argument('--folds', type=int, default=4,
                        help='set the number of folds')

    args = parser.parse_args()
    epoch = args.epoch
    folds = args.folds

    average_results(epoch, folds)
