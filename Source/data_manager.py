from subject import Subject
from pipelines import Data_Pipeline
import torch
from torch.utils.data import TensorDataset, ConcatDataset


class Data_Manager:

    def __init__(self, subjects_num: list, data_pipeline: Data_Pipeline):
        self.subjects = [Subject(num, data_pipeline) for num in subjects_num]
        self.data_pipeline = data_pipeline

    def data_info(self):
        """print the data info (subjects, sessions, positions, etc.)"""
        pass

    def get_dataset(self, experiments: list, include_synthetics = False) -> ConcatDataset:
        """extract a dataset of the given experiments from the main database

        experiments: list of strings in the template of 'subject_session_position' use * in one of the fields to
        indicate all
        include_synthetics: boolean, declare inclusion of synthetic data in the dataset"""
        datasets_lists = [subject.get_datasets(experiments, include_synthetics) for subject in self.subjects]
        datasets = [dataset for datasets in datasets_lists for dataset in datasets]  # flatten datasets list
        return ConcatDataset(datasets)
