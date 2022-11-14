import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from .pipelines import Data_Pipeline

# we need to consider memory issues here since we are going to work with large files, and create multiple objects.
# this might be a problem when we are going to work with the whole database.
# we need to think how we are going to handle this issue. a possible solution is to save the segments and features in
# files and load them when needed. this way we can work with the whole database without memory issues.
# another solution is to generate the objects on the fly, and not save them (implement in Data Manager).
# this way we can work with the whole database without memory issues.
# we need to think about the pros and cons of each solution (feel free to add more):
# saving the segments and features in files:
#   pros:
#       1. we can work with the whole database without memory issues.
#       2. we process the data once, and then we can use it as many times as we want.
#   cons:
#       1. we need to implement a mechanism to save and fetch the data from the files.
# generate data on the fly:
#   pros:
#       1. the implementation is simpler and pretty straight forward (using generators).
#   cons:
#       1. we need to process the data every time we want to use it.
#
# IMO the second solution is currently better, since we are still in the development stage, and we need to change
# the code a lot. once we are done with the development, we can implement the first solution. this way when we will
# start running lots of experiments we will not need to process the data every time.


class Recording:
    def __init__(self, file_path: str, data_pipeline: Data_Pipeline):
        self.file_path = file_path
        self.experiment = self.file_path_to_experiment(file_path)  # str, "subject_session_position" template
        self.data_pipeline = data_pipeline  # stores all preprocessing parameters
        self.signal = []  # np.array, the signal
        self.annotations = [] # list[(float, str], the annotations, time and description
        self.segments = []  # list[(segment: np.array, label)]
        self.subsegments = []  # np.array, stacked on dim 0
        self.features = []  # np.arrays, stacked on dim 0
        self.labels = []  # np.array of strings

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

    def get_dataset(self, include_synthetics = False) -> (np.array, np.array):
        """extract a dataset of the given recording file"""
        data = self.features
        labels = self.labels

        if include_synthetics:
            synth_subsegments, synth_labels = self.create_synthetic_subsegments()
            synth_features = self.extract_features(synth_subsegments)
            data = np.concatenate((data, synth_features), axis = 0)
            labels = np.concatenate((labels, synth_labels), axis = 0)
        return data, labels

    def match_experiment(self, experiment: str) -> bool:
        """check if the experiment matches the recording file"""
        rec_exp = self.experiment.split('_')
        curr_exp = experiment.split('_')
        if curr_exp[0] == rec_exp[0] or curr_exp[0] == '*':
            if curr_exp[1] == rec_exp[1] or curr_exp[1] == '*':
                if curr_exp[2] == rec_exp[2] or curr_exp[2] == '*':
                    return True
        return False

    def load_file(self, filename: str) -> (np.array, list[(float, str)]):
        """this function loads a file and returns the signal and the annotations
        in the future we might insert here the handling of merging files of part1 part2 (etc.) to one file"""
        signal = mne.io.read_raw_edf(filename, preload = True, stim_channel = 'auto').load_data().get_data()
        annotations = mne.read_annotations(filename)  # get annotations object from the file
        annotations = [(onset, description) for onset, description in zip(annotations.onset, annotations.description)
                       if 'Start_' in description or 'Release_' in description]
        self.signal, self.annotations = signal, annotations

    @staticmethod
    def file_path_to_experiment(file_path) -> str:
        """extract the experiment from the file path
        Currently the file name is in the form of: 'GR_pos#_###_S#_Recording_00_SD_edited.edf'
        and we want to extract the experiment in the form of: 'subject_session_position'"""
        file_name_parts = file_path.split('\\')[-1].split('_')
        for name in file_name_parts:
            if name[0] == 'S' and name[1] != 'D':
                session = name[1:]
            elif name[0:3] == 'pos':
                position = name[3:]
            elif len(name) == 3:
                subject = name
        try:
            experiment = f'{subject}_{session}_{position}'
        except NameError:
            print(f'Error: could not extract experiment from file path: {file_path}')
            experiment = ''
        return experiment
