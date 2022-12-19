from .paths_handler import Paths_Handler
from .recording import Recording
from .pipelines import Data_Pipeline
from torch.utils.data import TensorDataset, ConcatDataset
from pathlib import Path
import numpy as np


class Subject:

    def __init__(self, subject_num: int, data_pipeline: Data_Pipeline):
        self.subject_num = subject_num
        self.data_pipeline = data_pipeline
        self.recordings = self.load_recordings(load_data=False)

    def load_recordings(self, load_data=True):
        """load the recordings from the files"""
        files = self.my_files()
        files = self.experiment_files(files)
        recordings = [Recording(paths, self.data_pipeline) for paths in files]
        if load_data:
            [rec.load_file() for rec in recordings]
        return recordings

    @staticmethod
    def experiment_files(files: list[Path]) -> list[list[Path]]:
        """group the files by experiment, some experiment has several files, each has a different part of the
        experiment"""
        files_by_exp = []
        for file in files:
            if 'part' not in file.stem:
                files_by_exp.append([file])
                files.remove(file)

        names = []
        for file in files:
            names.append(file.stem.split('_part')[0])
        names = list(set(names))
        for name in names:
            files_by_exp.append([file for file in files if name in file.stem])

        return files_by_exp



    def my_files(self) -> list[Path]:
        """This function adds the paths of the subjects to the paths list"""
        paths_handler = Paths_Handler(self.data_pipeline.base_data_files_path)
        paths_handler.add_paths_of_subjects_num(self.subject_num)
        return paths_handler.paths

    def get_my_experiments(self, experiments: list[str] | str) -> list[str]:
        """extract the experiments that are in the subject"""
        if isinstance(experiments, str):
            experiments = [experiments]

        my_experiments = [rec.experiment for rec in self.recordings if
                          any([rec.match_experiment(exp) for exp in experiments])]
        return my_experiments

    def get_datasets(self, experiments: list | str, include_synthetics: bool = False) -> (np.array, np.array):
        """extract a dataset of the given experiments from the subject"""
        if isinstance(experiments, str):
            experiments = [experiments]

        # TODO: create a progress bar for this loop
        datasets = [rec.get_dataset(include_synthetics) for rec in self.recordings if
                    any([rec.match_experiment(exp) for exp in experiments])]

        if len(datasets) > 0:
            data = [data for data, labels in datasets]
            labels = [labels for data, labels in datasets]

            data = np.concatenate(tuple(data), axis=0)
            labels = np.concatenate(tuple(labels), axis=0)
        else:
            data = None
            labels = None
        return data, labels
