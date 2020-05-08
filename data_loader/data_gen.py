import numpy as np
import random
import utilities as util


class GenerateData:
    def __init__(self, train_labels, dev_labels, train_feat, dev_feat,
                 train_loc, dev_loc, train_indexes, dev_indexes, logger, config,
                 checkpoint):
        """
        Class which acts as a dataloader. Takes in training or validation
        data an outputs a generator of size equal to the specified batch.
        Information is recorded such as the current organisation state of the
        files in order to save for checkpointing. Also calculates the mean
        and standard deviations for the training data in order to normalise

        Inputs:
            train_labels: Labels for the training data including the indexes
            dev_labels: Labels for the validation data including the indexes
            train_feat: Array of training features
            dev_feat: Array of validation features
            train_loc: Location list of the length of all the files
            dev_loc: Location list of the length of all the files
            train_indexes: Indexes for the training data
            dev_indexes: Indexes for the validation data
            logger: Logger to record important information
            config: config file holding state information
            checkpoint: Bool - If true, load initial conditions from last
                        checkpoint
        """
        self.zeros_index_train = train_indexes[0]
        self.ones_index_train = train_indexes[1]
        self.zeros_index_dev = dev_indexes[0]
        self.ones_index_dev = dev_indexes[1]
        self.config = config
        self.audio_mode_is_concat_not_shorten = self.config.EXPERIMENT_DETAILS[
            'AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']
        self.batch_size = self.config.EXPERIMENT_DETAILS['BATCH_SIZE']
        self.feature_dim = self.config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
        self.convert_to_image = self.config.EXPERIMENT_DETAILS['CONVERT_TO_IMAGE']
        self.feature_experiment = self.config.EXPERIMENT_DETAILS['FEATURE_EXP']
        self.make_equal = self.config.EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']
        self.freq_bins = self.config.EXPERIMENT_DETAILS['FREQ_BINS']
        self.learning_procedure_train = self.config.LEARNING_PROCEDURE_TRAIN
        self.learning_procedure_dev = self.config.LEARNING_PROCEDURE_DEV
        self.train_labels = train_labels
        self.dev_labels = dev_labels
        self.train_feat = train_feat
        self.dev_feat = dev_feat
        self.train_loc = train_loc
        self.dev_loc = dev_loc
        self.logger = logger
        self.checkpoint = checkpoint

        if self.learning_procedure_train == 'whole_file':
            stats = 0
            for i, data in enumerate(self.train_loc):
                stats = np.hstack((self.train_feat[i][0:data[-1]]))
        else:
            stats = self.train_feat

        self.mean, self.standard_deviation = self.calculate_stats(stats)

    def calculate_stats(self, x):
        """
        Calculates the mean and the standard deviation of the input

        Input:
            x: Input data array

        Outputs
            mean: The mean of the input
            standard_deviation: The standard deviation of the input
        """
        if x.ndim == 1:
            mean = np.mean(x)
            standard_deviation = np.std(x)
            return mean, standard_deviation
        elif x.ndim == 2:
            axis = 0
        elif x.ndim == 3:
            axis = (0, 2)
        elif x.ndim == 4:
            axis = (0, 1, 3)

        mean = np.mean(x, axis=axis)
        mean = np.reshape(mean, (-1, 1))

        standard_deviation = np.std(x, axis=axis)
        standard_deviation = np.reshape(standard_deviation, (-1, 1))

        return mean, standard_deviation

    def resolve_batch_difference(self, batch_indexes, half_batch, indexes):
        """
        For imbalanced datasets, one class will be traversed more quickly
        than another and therefore will need to be reshuffled and potentially
        the batch will need to be updated from this re-shuffle.

        Inputs:
            batch_indexes: The indexes of the batch used to calculate the
                           current number of files in the batch
            half_batch: Used to determine how many examples should be in a
                        batch
            indexes: List of all the indexes for a particular class

        Outputs:
            batch_indexes: Updated array of a full batch index
            indexes: Shuffled list of indexes for the particular class
            pointer: Used to determine the position to begin sampling the
                     next batch of indexes
        """
        if batch_indexes.shape[0] != half_batch:
            current_length = batch_indexes.shape[0]
            difference = half_batch - current_length
            pointer = 0
            random.shuffle(indexes)
            temp = indexes[pointer:pointer + difference]
            pointer += difference
            batch_indexes = np.concatenate((batch_indexes,
                                            temp), axis=0)
            return batch_indexes, indexes, pointer

    def max_loc_diff(self, locators):
        """
        Used to calculate the longest file in the current experiment

        Input
            locators: List of start and end positions for each file in dataset

        Output
            max_value: The longest file length
            locators: A single vector of the file lengths
        """
        locators = np.array(locators)
        locators = locators[:, 1] - locators[:, 0]
        max_value = np.max(locators)

        return max_value, locators

    def criteria(self):
        """
        Determines whether the current configuration is make the dataset
        equal or not

        Output
            Bool whether the experiment is using make dataset equal or not
        """
        if self.make_equal:
            if self.learning_procedure_train == 'random_sample':
                return False
            if self.audio_mode_is_concat_not_shorten and \
                    self.learning_procedure_train == 'chunked_file':
                return False
            return True
        else:
            return False

    def generate_train_data(self, epoch, data_saver):
        """
        Generates the training batches to be used as inputs to a neural
        network. There are different ways of processing depending on whether
        the experiment is configured for random sampling of the data, chunked
        sampling (for instance 30s worth of data) or using the whole file. A
        generator is created which will be used to obtain the batch data and
        important information is also saved in order to load the experiment
        from a check point.

        Inputs
            epoch: The starting epoch
            data_saver: Dictionary to save the information for checkpoints

        Output
            batch_data: Current batched data for training
            batch_labels: Current labels associated to the batch data
            epoch: The current epoch
            reset: Bool - If a new epoch has been reached reset is set to True
            locs_array: Array of the length of each file in the batch
            data_saver: Dictionary containing information to be saved for
                        checkpoint saving
        """
        if self.learning_procedure_train == 'chunked_file' or \
                self.learning_procedure_train == 'whole_file':
            files_in_dataset = len(self.train_loc)
        else:
            files_in_dataset = len(self.train_labels[0])
        train_classes = np.array(self.train_labels[1])
        feature_dim = self.config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']

        if self.checkpoint:
            pointer_zero = data_saver['pointer_zero']
            pointer_one = data_saver['pointer_one']
            train_indexes_zeros = data_saver['index_zeros']
            train_indexes_ones = data_saver['index_ones']
        else:
            pointer_zero = pointer_one = 0
            train_indexes_zeros = np.array(self.zeros_index_train)
            train_indexes_ones = np.array(self.ones_index_train)
            random.shuffle(train_indexes_zeros)
            random.shuffle(train_indexes_ones)

        half_batch = self.batch_size // 2
        while True:
            reset = False
            # TODO For Debugging - Change Back
            proceed_make_equal = self.criteria()
            if proceed_make_equal:
                if pointer_one+half_batch >= files_in_dataset//2:
                    epoch += 1
                    reset = True

                batch_indexes_zeros = train_indexes_zeros[
                                      pointer_zero:pointer_zero+half_batch]
                batch_indexes_ones = train_indexes_ones[
                                     pointer_one:pointer_one+half_batch]

                pointer_zero += half_batch
                pointer_one += half_batch
            else:
                batch_indexes_zeros = train_indexes_zeros[
                                      pointer_zero:pointer_zero+half_batch]
                batch_indexes_ones = train_indexes_ones[
                                     pointer_one:pointer_one+half_batch]

                pointer_one += half_batch

                if pointer_zero >= train_indexes_zeros.shape[0]:
                    if batch_indexes_zeros.shape[0] != half_batch:
                        batch_indexes_zeros, train_indexes_zeros, \
                        pointer_zero = self.resolve_batch_difference(batch_indexes_zeros,
                                                                     half_batch,
                                                                     train_indexes_zeros)
                    else:
                        pointer_zero = 0
                else:
                    pointer_zero += half_batch

                if pointer_one >= train_indexes_ones.shape[0]:
                    epoch += 1
                    reset = True
                    if batch_indexes_ones.shape[0] != half_batch:
                        batch_indexes_ones, train_indexes_ones, pointer_one = \
                            self.resolve_batch_difference(batch_indexes_ones,
                                                          half_batch,
                                                          train_indexes_ones)
                    else:
                        pointer_one = 0
                    data_saver = {'pointer_one': pointer_one,
                                  'pointer_zero': pointer_zero,
                                  'index_ones': train_indexes_ones,
                                  'index_zeros': train_indexes_zeros}

            current_indexes = np.concatenate((batch_indexes_zeros,
                                              batch_indexes_ones),
                                             axis=0)

            if self.learning_procedure_train == 'chunked_file' or \
                    self.learning_procedure_train == 'whole_file':
                locs = [self.train_loc[i] for i in current_indexes.tolist()]
                max_value, locs_array = self.max_loc_diff(locs)
                current_batch_size = current_indexes.shape[0]
                if self.convert_to_image:
                    batch_data = np.zeros((current_batch_size*max_value,
                                           3,
                                           self.freq_bins,
                                           feature_dim))
                else:
                    batch_data = np.zeros((current_batch_size*max_value,
                                           self.freq_bins,
                                           feature_dim))

                for p, i in enumerate(locs):
                    temp = self.train_feat[i[0]:i[1]]
                    placeholder = p
                    for j in temp:
                        if self.convert_to_image:
                            batch_data[placeholder, :, :, :] = j
                        else:
                            batch_data[placeholder, :, :] = j
                        placeholder += current_batch_size
            else:
                batch_data = self.train_feat[current_indexes]
                locs_array = 0

            batch_labels = train_classes[current_indexes.tolist()]
            batch_data = util.normalise(batch_data, self.mean,
                                        self.standard_deviation)

            if reset and self.make_equal:
                pointer_zero = pointer_one = 0
                random.shuffle(train_indexes_zeros)
                random.shuffle(train_indexes_ones)
                data_saver = {'pointer_one': pointer_one,
                              'pointer_zero': pointer_zero,
                              'train_index_ones': train_indexes_ones,
                              'train_index_zeros': train_indexes_zeros}

            yield batch_data, batch_labels, epoch, reset, locs_array, data_saver

    def generate_development_data(self, epoch):
        """
        Generates the validation batches to be used as inputs to a neural
        network. There are different ways of processing depending on whether
        the experiment is configured for random sampling of the data, chunked
        sampling (for instance 30s worth of data) or using the whole file. A
        generator is created which will be used to obtain the batch data and
        important information is also saved in order to load the experiment
        from a check point.

        Inputs
            epoch: The current epoch used to start validation from a checkpoint

        Output
            batch_data: Current batched data for training
            batch_labels: Current labels associated to the batch data
            batch_folders: Current folders associated to the batch data
            locs_array: Array of the length of each file in the batch
        """
        folders = np.array(self.dev_labels[0])
        classes = np.array(self.dev_labels[1])
        dev_indexes_zeros = np.array(self.zeros_index_dev)
        dev_indexes_ones = np.array(self.ones_index_dev)
        indexes = np.concatenate((dev_indexes_zeros, dev_indexes_ones), axis=0)
        pointer = 0
        indexes = indexes.tolist()
        if self.checkpoint:
            for i in range(epoch):
                random.shuffle(indexes)
        else:
            random.shuffle(indexes)

        if self.learning_procedure_dev == 'whole_file' or \
                self.learning_procedure_dev == 'chunked_file':
            if len(self.dev_loc) % self.batch_size == 0:
                dev_counter = (len(self.dev_loc) // self.batch_size)
            else:
                dev_counter = (len(self.dev_loc) // self.batch_size) + 1
            for i in range(dev_counter):
                batch_indexes = indexes[pointer:pointer+self.batch_size]
                pointer += self.batch_size
                locs = [self.dev_loc[inter] for inter in batch_indexes]
                max_value, locs_array = self.max_loc_diff(locs)
                current_batch_size = locs_array.shape[0]
                if self.convert_to_image:
                    if self.feature_experiment == 'MFCC_concat':
                        batch_data = np.zeros((max_value*current_batch_size,
                                               self.freq_bins*3,
                                               self.feature_dim))
                    else:
                        batch_data = np.zeros((max_value*current_batch_size,
                                               3,
                                               self.freq_bins,
                                               self.feature_dim))
                else:
                    batch_data = np.zeros((max_value*current_batch_size,
                                           self.freq_bins,
                                           self.feature_dim))
                # batch_data = batch_data - 1e-5
                for p, j in enumerate(locs):
                    interim_data = self.dev_feat[j[0]:j[1]]
                    placeholder = p
                    for k in interim_data:
                        if self.convert_to_image and not \
                                self.feature_experiment == 'MFCC_concat':
                            batch_data[placeholder, :, :, :] = k
                        else:
                            batch_data[placeholder, :, :] = k
                        placeholder += current_batch_size
                batch_labels = classes[batch_indexes]
                batch_folders = folders[batch_indexes]
                batch_data = util.normalise(batch_data, self.mean,
                                            self.standard_deviation)

                yield batch_data, batch_labels, batch_folders, locs_array
        elif self.learning_procedure_dev == 'random_sample':
            while pointer < self.dev_feat.shape[0]:
                batch_indexes = indexes[pointer:pointer+self.batch_size]
                batch_data = self.dev_feat[batch_indexes]
                batch_labels = classes[batch_indexes]
                batch_folders = folders[batch_indexes]
                batch_data = util.normalise(batch_data, self.mean,
                                            self.standard_deviation)
                pointer += self.batch_size
                locs_array = np.ones((batch_data.shape[0]), dtype=np.int)

                yield batch_data, batch_labels, batch_folders, locs_array
