from .paths_handler import Paths_Handler
from .recording import Recording
from .pipelines import data_files_folder, Data_Pipeline
from .wrappers import reset
from torch.utils.data import TensorDataset, ConcatDataset


class Subject:

    def __init__(self, subject_num: int, data_pipeline: Data_Pipeline):
        self.subject_num = subject_num
        self.data_pipeline = data_pipeline
        self.files = Paths_Handler(data_files_folder).add_paths_of_subjects_num(subject_num).paths
        self.recordings = (Recording(file, data_pipeline) for file in self.files)

    def _reset(self):
        """reset the recordings generator"""
        self.recordings = (Recording(file, self.data_pipeline) for file in self.files)

    @reset
    def get_my_experiments(self, experiments: list) -> list[str]:
        """extract the experiments that are in the subject"""
        my_experiments = [rec.experiment for rec in self.recordings if any([rec.match_experiment(exp) for exp in experiments])]
        return my_experiments

    @reset
    def get_datasets(self, experiments: list, include_synthetics = False) -> list[TensorDataset]:
        """extract a dataset of the given experiments from the subject"""
        datasets = [rec.get_dataset(include_synthetics) for rec in self.recordings if any([rec.match_experiment(exp) for exp in experiments])]
        return datasets




