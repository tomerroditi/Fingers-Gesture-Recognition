from Paths_Handler import Paths_Handler
from recording import Recording
from pipelines import data_files_folder
from torch.utils.data import TensorDataset, ConcatDataset
from pipelines import Data_Pipeline


class Subject:

    def __init__(self, subject_num: int, data_pipeline: Data_Pipeline):
        self.subject_num = subject_num
        self.data_pipeline = data_pipeline
        self.files = Paths_Handler(data_files_folder).add_paths_of_subjects_num(subject_num).paths
        self.recordings = [Recording(file, data_pipeline) for file in self.files]

    def get_datasets(self, experiments: list, include_synthetics = False) -> list[TensorDataset]:
        """extract a dataset of the given experiments from the subject"""
        datasets = [rec.dataset for rec in self.recordings if rec.experiment in experiments]
        return datasets



