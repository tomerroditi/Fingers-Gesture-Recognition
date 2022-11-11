from .subject import Subject
from .pipelines import Data_Pipeline
from .wrappers import reset
from torch.utils.data import TensorDataset, ConcatDataset
import numpy as np


class Data_Manager:

    def __init__(self, subjects_num: list, data_pipeline: Data_Pipeline):
        self.subjects_num = subjects_num
        self.subjects = (Subject(num, data_pipeline) for num in subjects_num)
        self.data_pipeline = data_pipeline

    def _reset(self):
        """reset the subjects generator"""
        self.subjects = (Subject(num, self.data_pipeline) for num in self.subjects_num)

    @reset
    def data_info(self):
        """print the data info (subjects, sessions, positions, etc.)"""
        all_notation = ['*_*_*']
        experiments_in_datasets = [subject.get_my_experiments(all_notation) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list
        print(f'Experiments in datasets: {experiments_in_datasets}')

    @reset
    def get_dataset(self, experiments: str | list, include_synthetics = False) -> (ConcatDataset, list[str]):
        """
        extract a dataset of the given experiments from the main database

        experiments: list of strings in the template of 'subject_session_position' use * in one of the fields to
        indicate all. e.g. ['001_1_*', '002_*_*', '003_*_1']
        include_synthetics: boolean, declare inclusion of synthetic data in the dataset
        """
        experiments = list(experiments)  # make sure it's in list format

        experiments_in_datasets = [subject.get_my_experiments(experiments) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list

        datasets_lists = [subject.get_datasets(experiments, include_synthetics) for subject in self.subjects]
        datasets = [dataset for datasets in datasets_lists for dataset in datasets]  # flatten list

        data = np.concatenate(tuple([data for data, labels in datasets]), axis = 0)
        labels = np.concatenate(tuple([labels for data, labels in datasets]), axis = 0)

        return data, labels, experiments_in_datasets
