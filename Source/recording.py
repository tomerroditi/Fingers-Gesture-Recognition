import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from pipelines import Data_Pipeline

# we need to consider memory issues here since we are going to work with large files, and create multiple objects.
# this might be a problem when we are going to work with the whole database.
# we need to think how we are going to handle this issue. a possible solution is to save the segments and features in
# files and load them when needed. this way we can work with the whole database without memory issues.

class Recording:
    def __init__(self, file_path: str, data_pipeline: Data_Pipeline):
        self.file_path = file_path
        self.experiment = []  # str, "subject_session_position" - to be extracted from the file path
        self.data_pipeline = data_pipeline  # stores all preprocessing parameters
        self.signal, self.annotations = load_file(file_path)
        self.segments = []  # list of tuples (segment: np.array, label)
        self.subsegments = []  # list of strings
        self.labels = []  # list of strings
        self.features = []  # np.arrays
        self.dataset = []   # torch dataset

    def filter_signal(self):
        """filter the signal according to the data pipeline"""
        pass

    def segment_data(self):
        """discrete segmentation of the data according to the annotations"""
        # note that segments should be stacked on dim 0
        pass

    def make_subsegment(self):
        """make a subsegment of the segments"""
        # note that subsegments should be stacked on dim 0
        pass

    def create_synthetic_subsegments(self) -> (np.array, list[str]):
        """create synthetic data (subsegments) from the given recording file segments using HMM"""
        pass

    def preprocess_segments(self, segments: np.array) -> np.array:
        """preprocess the segments according to the data pipeline"""
        pass

    def extract_features(self, data: np.array) -> np.array:
        """extract features from the subsegments"""
        pass

    def create_dataset(self, include_synthetics = False) -> TensorDataset:
        """extract a dataset of the given recording file"""
        data = torch.Tensor(self.features)
        labels = torch.Tensor(self.labels)

        if include_synthetics:
            synth_subsegments, synth_labels = self.create_synthetic_subsegments()
            synth_features = self.extract_features(synth_subsegments)
            data = torch.cat((data, torch.Tensor(synth_features)), dim=0)
            labels = torch.cat((labels, torch.Tensor(synth_labels)))
        return TensorDataset(data, labels)


def load_file(filename: str) -> (np.array, list[tuple]):
    """this function loads a file and returns the signal and the annotations
    in the future we might insert here the handling of merging files of part1 part2 (etc.) to one file"""
    signal = mne.io.read_raw_edf(filename, preload = True, stim_channel = 'auto').load_data().get_data()
    annotations = mne.read_annotations(filename)  # get annotations object from the file
    annotations = [(onset, description) for onset, description in zip(annotations.onset, annotations.descripiton)
                   if 'Start' in description or 'Release' in description]
    return signal, annotations
