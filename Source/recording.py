import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from .pipelines import Data_Pipeline
from math import floor
from sklearn.cross_validation import train_test_split
from hmmlearn import hmm

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
        self.actions_list = []
        self.file_path = file_path
        self.experiment = self.file_path_to_experiment(file_path)  # str, "subject_session_position" template
        self.data_pipeline = data_pipeline  # stores all preprocessing parameters
        self.raw_signal = []  # RawEDF, the all raw signal without processing
        self.EMG_channels = []  # RawEDF, only the EMG signal, after filter
        self.acc_channels = []  # RawEDF, only the acc signal
        self.annotations = []  # list[(float, str], the annotations, time and description
        self.segments = []  # list[(segment: np.array, label)]
        self.subsegments = []  # np.array, stacked on dim 0
        self.features = []  # np.arrays, stacked on dim 0
        self.labels = []  # np.array of strings

    def filter_signal(self, frequencies):
        """filter the signal according to the data pipeline (currently only the EMG)"""
        self.EMG_channels.notch_filter(freqs=frequencies[0])
        self.EMG_channels.filter(l_freq=frequencies[1], h_freq=frequencies[2], method='iir')

    def separate_accelerometer_channels(self):
        """separate accelerometer data from EMG channels data"""
        self.acc_channels = self.raw_signal.copy().pick_channels(
            {"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
        self.EMG_channels = self.raw_signal.copy().drop_channels(
            {"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})

    def segment_data(self, signal):
        """discrete segmentation of the data according to the annotations"""
        # note that segments should be stacked on dim 0 - need to ask Tomer:
        # we got 10X10 (10 repetition for 10 actions) numpy array
        # where each cell is in the form of 19Xi=num_of_samples

        signal.set_index('time', drop=True, inplace=True)
        annotations_df = pd.DataFrame(self.annotations)
        actions = np.array([action[6:] for action in
                            annotations_df[annotations_df[1].str.contains('Start')][1].unique()])
        data_by_actions = []
        for action in actions:
            action_times = annotations_df[annotations_df[1].str.contains(action)][0].reset_index(drop=True)
            selected_rows = [signal.loc[action_times[i]:action_times[i + 1]].to_numpy() for i in
                             range(0, len(action_times), 2)]
            self.segments.append(np.array(selected_rows))
        self.actions_list = actions

    @staticmethod
    def make_subsegment(window_len, action_data):
        """make a subsegment of the segments"""
        # note that subsegments should be stacked on dim 0
        # ! window len = number of rows, meaning we need to convert it from sec/samples

        action_data = [np.array_split(repetition[:window_len * floor(repetition.shape[0] / window_len)],
                                      floor(repetition.shape[0] / window_len)) for repetition in action_data]
        return action_data

    @staticmethod
    def split_data_for_synthetic(action):
        # they split the val from the train, here is different
        flat_list = np.array([item for sublist in action for item in sublist])
        action_train, action_test = train_test_split(flat_list, test_size=0.5)
        action_test, action_val = train_test_split(action_test, test_size=100/3)
        return action_train, action_test, action_val

    def create_synthetic_subsegments(self, comp_num, iterations_num) -> (np.array, list[str]):
        # not finished, need to test before.
        # they not used test and validation data

        """create synthetic data (subsegments) from the given recording file segments using HMM"""
        features_train, features_test, features_val, label_train, label_test, label_val = []
        for i in range(0,len(self.features)):
            [action_train, action_test, action_val] = self.split_data_for_synthetic(self, self.features[i])
            features_train.append(action_train)
            features_test.append(action_test)
            features_val.append(action_val)
            label_train.append([self.actions_list[i]]*action_train.shape[0])  # need to check dimension
            label_test.append([self.actions_list[i]]*action_test.shape[0])  # need to check dimension
            label_val.append([self.actions_list[i]]*features_val.shape[0])  # need to check dimension

        models = []
        for i in range(len(self.actions_list)):
            #X = np.concatenate([train_data[repetitions*i+j,:,:] for j in [0,1,2,3,4,5]]) - why??
            model = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=10)
            lengths = [features_train.shape[1] for j in range(6)]  # need to check dimension, the sum of this sould be n_sample. doesnt make sense
            # x should be in the form of (n_samples, n_features=16)
            model.fit(features_train, lengths)
            models.append(model)
        test_res = []
        for s in range(len(self.actions_list) * 2):
            test_res.append(np.argmax(np.array([models[i].score(features_test[s, :, :]) for i in range(len(self.actions_list))])))

        acc = np.mean(np.array(test_res) == label_test) # mean of all models

        generated_data, generated_labels = []
        for i in range(len(self.actions_list)):
            for j in range(6):
                # they loaded model from file, maybe is eas the best one from the trials.
                X, Z = model.sample(11) # should decide if 11 (supposed to be min window)
                generated_data.append(X)
            labels = [self.actions_list[i]] * (6*11)  # how they know that it match this action?
            generated_labels.append(labels)

        generated_data = np.concatenate((features_train, generated_data))
        generated_labels = np.concatenate((label_train.flatten(), generated_labels.flatten())) # check dimensions, if flatten needed

    def preprocess_segments(self, frequencies, window_len) -> np.array:
        """preprocess the segments according to the data pipeline"""

        # apply filter on the EMG channels (later it will also be done on the acc)
        self.separate_accelerometer_channels(self)
        self.filter_signal(frequencies)
        filtered_signal = self.acc_channels.to_data_frame().merge(self.EMG_channels.to_data_frame())

        # first segmentation of the all signal - by full movements
        self.segment_data(filtered_signal)

        # second segmentation - each action (& repetition) by window length
        for action in self.segments:
            self.subsegments.append(self.make_subsegment(window_len, action)) # maybe will need to change to np.append
            self.subsegments = np.array(self.subsegments)

        for action in self.subsegments:
            self.features.append((self.extract_features(action)))

    @staticmethod
    def extract_features(self, data: np.array, use_acc=False) -> np.array:
        """extract features from the subsegments"""
        # calculate RMS + normalize on each of the subsegments
        for repetition in data:
            final_subsegment = []
            if ~use_acc:
                repetition = [subsegment[:, 3:] for subsegment in repetition]
            subsegment_rms = [np.sqrt(np.mean(subsegment[:] ** 2,axis=0)) for subsegment in repetition]
            subsegment_rms = subsegment_rms / np.max(np.absolute(subsegment_rms))
            final_subsegment.append(subsegment_rms)
        return final_subsegment

    def get_dataset(self, include_synthetics=False) -> (np.array, np.array):
        """extract a dataset of the given recording file"""
        data = self.features
        labels = self.labels

        if include_synthetics:
            synth_subsegments, synth_labels = self.create_synthetic_subsegments()
            synth_features = self.extract_features(synth_subsegments)
            data = np.concatenate((data, synth_features), axis=0)
            labels = np.concatenate((labels, synth_labels), axis=0)
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
        signal = mne.io.read_raw_edf(filename, preload=True, stim_channel='auto').load_data()
        annotations = mne.read_annotations(filename)  # get annotations object from the file
        annotations = [(onset, description) for onset, description in zip(annotations.onset, annotations.description)
                       if 'Start_' in description or 'Release_' in description]
        self.raw_signal, self.annotations = signal, annotations

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
