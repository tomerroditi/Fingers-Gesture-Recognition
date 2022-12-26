import collections
from hmmlearn import hmm
from .subject import Subject
from .pipelines import Data_Pipeline
from .wrappers import reset
import numpy as np


class Data_Manager:

    def __init__(self, subjects_num: list[int], data_pipeline: Data_Pipeline):
        self.subjects_num = subjects_num
        self.subjects = [Subject(num, data_pipeline) for num in subjects_num]
        self.data_pipeline = data_pipeline

    def data_info(self):
        """print the data info (subjects, sessions, positions, etc.)"""
        all_notation = ['*_*_*']
        experiments_in_datasets = [subject.get_my_experiments(all_notation) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list
        print(f'Experiments in datasets: {experiments_in_datasets}')
        print('Experiments format is: subject_session_position')

    def get_dataset(self, experiments: str or list, include_synthetics = False) -> (np.array, np.array):
        """
        extract a dataset of the given experiments from the main database

        experiments: list of strings in the template of 'subject_session_position' use * in one of the fields to
        indicate all. e.g. ['001_1_*', '002_*_*', '003_*_1']
        include_synthetics: boolean, declare inclusion of synthetic data in the dataset
        """
        if isinstance(experiments, str):
            experiments = [experiments]

        experiments_in_datasets = [subject.get_my_experiments(experiments) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list

        # TODO: create a progress bar for this loop
        datasets = [subject.get_datasets(experiments, include_synthetics) for subject in self.subjects]
        datasets = [dataset for dataset in datasets if dataset[0] is not None and dataset[1] is not None]

        data = np.concatenate(tuple([data for data, labels in datasets]), axis = 0)
        labels = np.concatenate(tuple([labels for data, labels in datasets]), axis = 0)

        return data, labels, experiments_in_datasets

    @staticmethod
    def add_synthetics(data: np.array, labels: np.array, num: int) -> (np.array, np.array):
        # todo: rename the num parameter to something more meaningful (i don't know what it means)
        """
        add synthetic data to the dataset, currently only hmm for feature generation is supported.
        this function has been removed from the recording object to prevent data leakage!
        inputs:
            data: np array of shape (n_samples, n_features)
            labels: string np array of shape (n_samples) in the format of:
                    '<experiment name>_<gesture name>_<gesture number>'
        outputs:
            data: np array of shape (n+m_samples, n_features)
            labels: string np array of shape (n+m_samples)
        """
        # place holders
        synthetic_data = np.empty((0, data.shape[1]))
        generated_labels = np.empty(0, dtype=str)
        # get labels without number of gesture
        labels_no_num = np.char.rstrip(labels, '_0123456789')
        # count how many gestures of each type we have (separate by experiment as well)
        labels_unique = np.unique(labels_no_num)
        labels_unique = np.char.rstrip(labels_unique, '_0123456789')
        labels_counter = collections.Counter(labels_unique)
        # create the synthetic data using hmm
        for label in np.unique(labels_no_num):
            curr_data = data[labels_no_num == label]
            model = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=10)
            lengths = [len(curr_data) // num for _ in range(num)]
            if sum(lengths) == 0:
                continue
            model.fit(curr_data[:sum(lengths)], lengths)
            for j in range(num):
                mean_num_windows = round(len(curr_data) / labels_counter[label])
                curr_synthetic_data, _ = model.sample(mean_num_windows)
                np.concatenate((synthetic_data, curr_synthetic_data), axis=0)
                curr_labels = np.array([label for _ in range(mean_num_windows)])
                np.concatenate((generated_labels, curr_labels), axis=0)

        # concatenate the synthetic data to the original data
        data = np.concatenate((data, synthetic_data), axis=0)
        labels = np.concatenate((labels, generated_labels), axis=0)
        return data, labels
