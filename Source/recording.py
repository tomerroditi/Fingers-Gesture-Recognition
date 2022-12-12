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
    def __init__(self, file_path: Path, pipeline: Data_Pipeline):
        self.file_path = file_path
        self.experiment = self.file_path_to_experiment(file_path)  # str, "subject_session_position" template
        self.pipeline = pipeline  # stores all preprocessing parameters
        self.raw_edf_emg = None  # mne.io.edf.edf.RawEDF object with emg channels only
        self.raw_edf_acc = None  # mne.io.edf.edf.RawEDF object with accelerometer channels only
        self.annotations = None  # list[(float, str], the annotations, time and description
        self.annotations_data = None  # list[(np.array, np.array, str)], EMG data, accelerometer data, gesture label
        self.segments = None  # list[(np.array, np.array)], EMG data, accelerometer data
        self.labels = None  # np.array(str), the labels of the segments
        self.features = None  # np.array, stacked on dim 0
        self.segments_labels = None  # np.array of strings

    @staticmethod
    def file_path_to_experiment(file_path: Path) -> str:
        """extract the experiment from the file path
        Currently the file name is in the form of: 'GR_pos#_###_S#_Recording_00_SD_edited.edf'
        and we want to extract the experiment in the form of: 'subject_session_position'"""
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
        filename = str(self.file_path)
        raw_edf = mne.io.read_raw_edf(filename, preload = True, stim_channel = 'auto', verbose = False)
        self.raw_edf_acc = raw_edf.copy().pick_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
        self.raw_edf_emg = raw_edf.copy().drop_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})

        # TODO: add a validation to the annotations, check that the annotations are in the right order - start, stop,
        #  start, stop, etc. and that each consecutive start, stop annotations are of the same gesture.
        annotations = mne.read_annotations(filename)  # get annotations object from the file
        self.annotations = [(onset, description) for onset, description in
                            zip(annotations.onset, annotations.description)
                            if 'Start_' in description or 'Release_' in description]

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
                emg_data = self.raw_edf_emg.get_data(tmin=start_time, tmax=end_time)
                acc_data = self.raw_edf_acc.get_data(tmin=start_time, tmax=end_time)
                annotations_data.append((emg_data, acc_data, label))
        return annotations_data

    def heatmap_visualization(self, features: np.array):
        # If the channels are not arranged in chronological order,
        # should get the location and rearrange them here.

        # it is better if we would use self.features instead of passing features into the function.
        heatmaps = [np.reshape(segment, (4, 4)) for segment in features]  # np iterates over dim 0
        gestures_num = len(self.labels.unique())
        repetitions = len(features) / gestures_num  # be more concise with how you call variables, it's not clear what
        # repetitions var is from its name. in other function it was used for the data of the gesture. this makes it
        # harder to follow the code. you could use num_repetitions instead (same with gesture_num -> num_gestures)
        fig, axs = plt.subplots(gestures_num, repetitions + 1)
        # you are assuming that the gestures are ordered by labels, and that each gesture has the same number of
        # repetitions. these are things we should avoid assuming cause in the future it's not likely to be true.
        # the same goes for how you assumed this in your segment data function which is why I think we should stick with
        # my implementation (of the segmentation) that is more general.
        # so for now this function might be fine, but we need to change it for a more general solution.
        for ac in range(len(heatmaps)):
            im = axs[int(ac / repetitions), ac % repetitions + 1].imshow(heatmaps[ac], cmap='hot')
        for i in range(gestures_num):
            # this is alot of code for a simple subplot. you can use fig = plt.figure, and then add subplots with
            # fig.add_subplot like this:
            # fig = plt.figure()
            # fig.add_subplot(num_gestures, num_repetitions, 1)
            # for i in range(num_gestures):
            #     for j in range(num_repetitions):
            #         ax = fig.add_subplot(num_gestures, num_repetitions, i * num_repetitions + j # index)
            #         ax.plot(the_data)  # any plotting type you need here
            #         ax.set_title(title), etc.
            # fig.show()
            # this way you are using the simplified API of matplotlib, and you can add more subplots easily without
            # having to set the size of the figure or any other annoying parameters.
            for j in range(repetitions + 1):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            axs[i, 0].text(-0.5, 0.5, self.labels.unique()[i], fontsize=18)
            axs[i, 0].spines['top'].set_visible(False)
            axs[i, 0].spines['right'].set_visible(False)
            axs[i, 0].spines['bottom'].set_visible(False)
            axs[i, 0].spines['left'].set_visible(False)
        cbar = fig.colorbar(im, ax=axs.ravel().tolist())
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
        generated_data = []
        generated_labels = []
        for action_name in self.labels.unique():
            # it better if you write these lines like this:
            # curr_action_features = self.features[self.labels == action_name]
            # [features_train, features_test, features_val] = self.split_data_for_synthetic(self, curr_action_features)
            # this is much more readable, and you don't need to use the index directly
            indices = [i for i, x in enumerate(self.labels) if x == action_name]
            [features_train, features_test, features_val] = self.split_data_for_synthetic(self.features[indices])
            # you can use the same trick here
            # curr_action_labels = self.labels[self.labels == action_name]
            # label_test.append(curr_action_labels)
            label_test.append(action_name * features_test.shape[0])
            model = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=10)
            lengths = [len(features_train)//6 for j in range(6)]  # it's better to use a variable instead of a number
            model.fit(features_train[:sum(lengths)], lengths)
            test_res.append(np.argmax(np.array(model.score(features_test))))
            for j in range(6):
                X, Z = model.sample(11)  # should decide if 11 (supposed to be min window)
                generated_data.append(X)
                generated_labels.append(action_name * (6 * 11))  # are you sure these are labels? im pretty sure this will prompt an error
        # why do we need this?
        acc = np.mean(np.array(test_res) == label_test)  # mean of all models on all the data
        return generated_data, generated_labels

    @staticmethod
    def split_data_for_synthetic(action_features: np.array):
        """spilt action's features into 50% train, 33.3% test and 16.6% validation sets"""
        action_train, action_test = train_test_split(action_features, test_size=0.5)
        action_test, action_val = train_test_split(action_test, test_size=1 / 3)  # we are not using the validation set, so why do we need it?
        return action_train, action_test, action_val

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
