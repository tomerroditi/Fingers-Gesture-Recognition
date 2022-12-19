from .subject import Subject
from .pipelines import Data_Pipeline
from .wrappers import reset
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np


class Data_Manager:

    def __init__(self, subjects_num: list[int], data_pipeline: Data_Pipeline):
        self.subjects_num = subjects_num
        self.subjects = [Subject(num, data_pipeline) for num in subjects_num]
        self.data_pipeline = data_pipeline

    def _reset(self):
        """reset the subjects generator"""
        self.subjects = (Subject(num, self.data_pipeline) for num in self.subjects_num)

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
