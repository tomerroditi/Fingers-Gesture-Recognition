import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .pipelines import Data_Pipeline
from pathlib import Path
from math import floor
from .feature_extractors import build_feature_extractor
import hmmlearn.hmm as hmm
from sklearn.model_selection import train_test_split
import mne
mne.set_log_level('WARNING')  # disable mne function's printing except warnings

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
    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        self.files_path = files_path
        self.experiment = self.file_path_to_experiment(files_path)  # str, "subject_session_position" template
        self.pipeline = pipeline  # stores all preprocessing parameters
        self.raw_edf_emg = None  # mne.io.edf.edf.RawEDF object with emg channels only
        self.raw_edf_acc = None  # mne.io.edf.edf.RawEDF object with accelerometer channels only
        self.annotations = None  # list[(float, str], the annotations, time and description
        self.annotations_data = None  # list[(np.array, np.array, str)], EMG data, accelerometer data, gesture label
        self.segments = None  # list[(np.array, np.array)], EMG data, accelerometer data
        self.labels = None  # np.array(str), the labels of the segments
        self.features = None  # np.array, stacked on dim 0
        self.segments_labels = None  # np.array of strings
        self.num_repetitions_per_gesture = {}  # dict of the number of repetitions of each gesture

    @staticmethod
    def file_path_to_experiment(files_path: list[Path]) -> str:
        """extract the experiment from the file path
        Currently the file name is in the form of: 'GR_pos#_###_S#_Recording_00_SD_edited.edf'
        and we want to extract the experiment in the form of: 'subject_session_position'"""
        file_path = files_path[0]  # we can take the exp from the first file, since all files are from the same exp
        file_name_parts = file_path.stem.split('_')
        for name in file_name_parts:
            if name[0].upper() == 'S' and name[1].upper() != 'D':
                session = name[1:]
            elif name[0:3].upper() == 'POS':
                position = name[3:]
            elif len(name) == 3:
                subject = name
        try:
            experiment = f'{subject}_{session}_{position}'
        except NameError:
            raise NameError(f'Error: could not extract experiment name from file path: {file_path}.'
                  f'pls check the file name format.')
        return experiment

    def load_file(self):
        """this function loads the files data and sets the raw_edf and annotations field.
        in the future we might insert here the handling of merging files of part1, part2, etc. to one file"""
        # TODO: add a validation to the annotations, check that the annotations are in the right order - start, stop,
        #  start, stop, etc. and that each consecutive start, stop annotations are of the same gesture.
        # TODO: we need to add raw files fixing as well (deleting the bad data, removing bad annotations, etc.)

        files_names = [str(path) for path in self.files_path]
        files_names.sort()  # part1, part2, etc. should be in order
        annotations = []
        for file_name in files_names:
            raw_edf = mne.io.read_raw_edf(file_name, preload = True, stim_channel = 'auto', verbose = False)
            curr_raw_edf_acc = raw_edf.copy().pick_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
            curr_raw_edf_emg = raw_edf.copy().drop_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
            if self.raw_edf_acc is None:
                self.raw_edf_acc = curr_raw_edf_acc
                self.raw_edf_emg = curr_raw_edf_emg
            else:
                self.raw_edf_acc.append(curr_raw_edf_acc)
                self.raw_edf_emg.append(curr_raw_edf_emg)

        annotations = self.raw_edf_emg.annotations
        annotations = [(onset, description) for onset, description in
                            zip(annotations.onset, annotations.description)
                            if 'Start_' in description or 'Release_' in description]

        for i, annotation in enumerate(annotations):
            annotations[i] = (annotation[0], annotation[1].rstrip(' 0123456789'))  # remove the number from the annotation

        self.annotations = annotations

    def preprocess_data(self) -> np.array:
        """preprocess the segments according to the data pipeline"""
        # TODO: consider converting all these methods to static methods, and then call them from here, this way we can
        #  test them separately and more conveniently, and we can use them in other places as well.
        if self.raw_edf_emg is None:
            self.load_file()

        self.raw_edf_emg, self.raw_edf_acc = self.filter_signal()
        self.annotations_data = self.extract_annotated_data()

        if self. pipeline.segmentation_type == "discrete":
            self.segments, self.labels = self.segment_data_discrete()
        elif self.pipeline.segmentation_type == "continuous":
            self.segments, self.labels = self.segment_data_continuous()

        # self.segments = self.normalize_segments()
        self.features = self.extract_features()
        # self.features = self.normalize_features()

    def filter_signal(self) -> (mne.io.edf.edf.RawEDF, mne.io.edf.edf.RawEDF):
        """filter the signal according to the data pipeline (currently only the EMG, acc will be added later here)"""
        raw_edf_emg = self.raw_edf_emg.notch_filter(freqs=self.pipeline.emg_notch_freq)
        raw_edf_emg = raw_edf_emg.filter(l_freq=self.pipeline.emg_low_freq, h_freq=self.pipeline.emg_high_freq
                                                   , method='iir')

        return raw_edf_emg, self.raw_edf_acc

    def extract_annotated_data(self) -> list[(np.array, np.array, str)]:
        """extract the data of the gestures. It might be used for exploring the data, or for discrete segmentation"""
        annotations_data = []
        for i, annotation in enumerate(self.annotations):
            time, description = annotation
            if 'Release_' in description:
                continue
            elif 'Start_' in description:
                start_time = time
                end_time = self.annotations[i + 1][0] # the next annotation is the end of the gesture
                label = description[6:]
                self.num_repetitions_per_gesture[label] = self.num_repetitions_per_gesture.get(label, 0) + 1
                emg_data = self.raw_edf_emg.get_data(tmin=start_time, tmax=end_time)
                acc_data = self.raw_edf_acc.get_data(tmin=start_time, tmax=end_time)
                annotations_data.append((emg_data, acc_data, label))
        return annotations_data

    def heatmap_visualization(self, data: np.array, num_gestures: int, num_repetitions_per_gesture: dict):
        # If the channels are not arranged in chronological order,
        # should get the location and rearrange them here.

        heatmaps = [np.reshape(segment, (4, 4)) for segment in data]  # np iterates over dim 0
        fig, axes = plt.subplots(num_gestures, max(num_repetitions_per_gesture.values()))
        idx = 0
        for i in range(num_gestures):
            for j in range(num_repetitions_per_gesture[self.labels.unique()[i]]):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                im = axes[i, j].imshow(heatmaps[idx], cmap='hot')
                idx += 1
            fig.axes[i*max(num_repetitions_per_gesture.values())].set_ylabel('h', rotation=0, fontsize=18)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.tick_params(labelsize=18)
        plt.show()

    def segment_data_discrete(self) -> ((np.array(np.float16), np.array(np.float16)), np.array(str)):
        """
        discrete segmentation of the data according to the annotations. this is to repeat last year results.
        use float16 dtype to save memory
        """
        segment_length_emg = floor(self.pipeline.segment_length_sec * self.pipeline.emg_sample_rate)
        segment_length_acc = floor(self.pipeline.segment_length_sec * self.pipeline.acc_sample_rate)

        segments_emg = np.empty(shape=(0, 16, segment_length_emg), dtype=np.float16)
        segments_acc = np.empty(shape=(0, 3, segment_length_acc), dtype=np.float16)
        labels = np.empty(shape=(0,), dtype=str)
        for emg_data, acc_data, label in self.annotations_data:
            # emg segmentation and labels creation
            for i in range(0, emg_data.shape[1] - segment_length_emg, segment_length_emg):
                curr_emg_segment = emg_data[:, i:i + segment_length_emg][np.newaxis, :, :]
                segments_emg = np.vstack((segments_emg, curr_emg_segment))
                labels = np.append(labels, label)
            # acc segmentation
            for i in range(0, acc_data.shape[1] - segment_length_acc, segment_length_acc):
                curr_acc_segment = acc_data[:, i:i + segment_length_acc][np.newaxis, :, :]  # need to add a new axis
                segments_acc = np.vstack((segments_acc, curr_acc_segment))

        segments = (segments_emg, segments_acc)
        return segments, labels

    def segment_data_continuous(self) -> ((np.array(np.float16), np.array(np.float16)), np.array(str)):
        raise NotImplementedError

    def normalize_segments(self) -> (np.array(np.float16), np.array(np.float16)):
        raise NotImplementedError

    def extract_features(self) -> np.array:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        features = feature_extractor.extract_features(self.segments, **self.pipeline.features_extraction_params)
        return features

    def normalize_features(self) -> np.array:
        raise NotImplementedError

    def get_dataset(self, include_synthetics = False) -> (np.array, np.array):
        """extract a dataset of the given recording file"""
        if self.features is None:
            self.preprocess_data()

        data = self.features
        labels = self.labels

        if include_synthetics:
            synth_data, synth_labels = self.create_synthetic_data()
            data = np.concatenate((data, synth_data), axis = 0)
            labels = np.concatenate((labels, synth_labels), axis = 0)
        return data, labels

    def create_synthetic_data(self) -> (np.array, list[str]):
        """create synthetic data (features) from the extracted features using HMM"""

        test_res = []
        label_test = []
        generated_data = np.empty((0, 1, 4, 4))
        generated_labels = np.empty(0, dtype=str)
        for action_name in np.unique(self.labels):
            curr_action_features = self.features[self.labels == action_name]
            curr_action_features = curr_action_features.squeeze()
            curr_action_features = curr_action_features.reshape(curr_action_features.shape[0], -1)
            [features_train, features_test] = self.split_data_for_synthetic(curr_action_features)
            label_test.append(action_name * features_test.shape[0])
            model = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=10)
            lengths = [len(features_train) // self.pipeline.num_repetition_hmm for j in range(self.pipeline.num_repetition_hmm)]
            model.fit(features_train[:sum(lengths)], lengths)
            # test_res.append(np.argmax(np.array(model.score(features_test))))
            for j in range(self.pipeline.num_repetition_hmm):
                mean_num_windows = round(len(curr_action_features) / self.num_repetitions_per_gesture[action_name])
                x, _ = model.sample(mean_num_windows)
                x = x.reshape(x.shape[0], 1, 4, 4)
                np.concatenate((generated_data, x), axis=0)
                labels = np.array([action_name for i in range(x.shape[0])])
                np.concatenate((generated_labels, labels), axis=0)
        # why do we need this?
        # acc = np.mean(list(np.array(test_res)) == label_test)  # mean of all models on all the data
        return generated_data, generated_labels

    @staticmethod
    def split_data_for_synthetic(action_features: np.array):
        """spilt action's features into 80% train, 20% test"""
        action_train, action_test = train_test_split(action_features, test_size=0.2)
        return action_train, action_test

    def match_experiment(self, experiment: str) -> bool:
        """check if the experiment matches the recording file"""
        rec_exp = self.experiment.split('_')
        curr_exp = experiment.split('_')

        # add zeros in case the subject num is less than 3 digits
        while len(curr_exp[0]) < 3 and curr_exp[0] != '*':
            curr_exp[0] = '0' + curr_exp[0]

        if curr_exp[0] == rec_exp[0] or curr_exp[0] == '*':
            if curr_exp[1] == rec_exp[1] or curr_exp[1] == '*':
                if curr_exp[2] == rec_exp[2] or curr_exp[2] == '*':
                    return True
        return False
