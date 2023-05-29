"""This module contains the data manager class and its helper classes. It is used to construct datasets from raw data
files using the data pipeline. The public API of the module is the Data_Manager class."""
import collections
import numpy as np
import matplotlib.pyplot as plt
import mne

from hmmlearn import hmm
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
from math import floor
from typing import TypeVar
from Source.streamer.data import Data

Data_Pipeline = TypeVar("Data_Pipeline")
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
class Data_Manager:

    def __init__(self, subjects_num: list[int], data_pipeline: Data_Pipeline):
        self.subjects_num = subjects_num
        self.subjects = [Subject(num, data_pipeline) for num in subjects_num]
        self.data_pipeline = data_pipeline

    def data_info(self):
        """print the data info (subjects, sessions, positions, etc.)"""
        all_notation = ['*_*_*']
        experiments_in_datasets = [subject.get_my_experiments(all_notation) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list
        print(f'Available experiments in the data manager: {experiments_in_datasets}')
        print('Experiments format is: subject_session_position')

    def get_dataset(self, experiments: str or list) -> (np.array, np.array):
        """
        extract a dataset of the given experiments from the main database

        experiments: list of strings in the template of 'subject_session_position' use * in one of the fields to
        indicate all. e.g. ['001_1_*', '002_*_*', '003_*_1']
        include_synthetics: boolean, declare inclusion of synthetic data in the dataset
        """
        if isinstance(experiments, str):
            experiments = [experiments]

        experiments_in_datasets = [subject.get_my_experiments(experiments) for subject in self.subjects]
        experiments_in_datasets = [exp for exp_list in experiments_in_datasets for exp in exp_list]  # flatten list
        print(f'Experiments in datasets: {experiments_in_datasets}\n'
              f'Starting to extract datasets')

        datasets = []
        for exp in tqdm(experiments_in_datasets, desc='Loading experiments datasets', unit='exp'):  # noqa
            for subject in self.subjects:
                if subject.get_my_experiments(exp):
                    datasets.append(subject.get_dataset(exp))
                    break
        print('finished extracting the dataset')

        try:
            data = np.concatenate(tuple([data for data, labels in datasets]), axis=0)
            labels = np.concatenate(tuple([labels for data, labels in datasets]), axis=0)
        except ValueError as e:
            if str(e) == 'need at least one array to concatenate':
                print('No data was found for the specified experiments')
                return None, None
            else:
                raise e

        return data, labels

    @staticmethod
    def add_synthetics(data: np.array, labels: np.array, num: int) -> (np.array, np.array):
        # todo: rename the num parameter to something more meaningful (i don't know what it means)
        """
        add synthetic data to the dataset, currently only hmm for feature generation is supported.
        this function has been removed from the recording object to prevent data leakage!
        inputs:
            data: np array of shape (n_samples, n_features)
            labels: string np array of shape (n_samples) in the format of:
                    '<experiment name>_<gesture name>_<gesture number>'
        outputs:
            data: np array of shape (n+m_samples, n_features)
            labels: string np array of shape (n+m_samples)
        """
        # place holders
        synthetic_data = np.empty((0, data.shape[1]))
        generated_labels = np.empty(0, dtype=str)
        # get labels without number of gesture
        labels_no_num = np.char.rstrip(labels, '_0123456789')
        # count how many gestures of each type we have (separate by experiment as well)
        labels_unique = np.unique(labels_no_num)
        labels_unique = np.char.rstrip(labels_unique, '_0123456789')
        labels_counter = collections.Counter(labels_unique)
        # create the synthetic data using hmm
        for label in np.unique(labels_no_num):
            curr_data = data[labels_no_num == label]
            model = hmm.GaussianHMM(n_components=4, covariance_type="tied", n_iter=10)
            lengths = [len(curr_data) // num for _ in range(num)]
            if sum(lengths) == 0:
                continue
            model.fit(curr_data[:sum(lengths)], lengths)
            for j in range(num):
                mean_num_windows = round(len(curr_data) / labels_counter[label])
                curr_synthetic_data, _ = model.sample(mean_num_windows)
                np.concatenate((synthetic_data, curr_synthetic_data), axis=0)
                curr_labels = np.array([label for _ in range(mean_num_windows)])
                np.concatenate((generated_labels, curr_labels), axis=0)

        # concatenate the synthetic data to the original data
        data = np.concatenate((data, synthetic_data), axis=0)
        labels = np.concatenate((labels, generated_labels), axis=0)
        return data, labels


class Subject:

    def __init__(self, subject_num: int, data_pipeline: Data_Pipeline):
        self.subject_num = subject_num
        self.data_pipeline = data_pipeline
        self.recordings = self.load_recordings(load_data=False)

    def load_recordings(self, load_data=True):
        """load the recordings from the files"""
        files = self.my_files()
        files = self.experiment_files(files)
        if self.data_pipeline.available_data == 'emg':
            recordings = [Recording_Emg(paths, self.data_pipeline) for paths in files]
        elif self.data_pipeline.available_data == 'emg_acc':
            recordings = [Recording_Emg_Acc(paths, self.data_pipeline) for paths in files]
        elif self.data_pipeline.available_data == 'emg_acc_gyro':
            raise NotImplementedError

        recordings = [Recording_Emg_Acc(paths, self.data_pipeline) for paths in files]
        if load_data:
            [rec.load_file() for rec in
             tqdm(recordings, desc='Loading recordings files', leave=False, unit='rec')]  # noqa
        return recordings

    def my_files(self) -> list[Path]:
        """This function adds the paths of the subjects to the paths list"""
        subject_num = f'{self.subject_num:03d}'  # convert to 3 digits string format
        # folders of specified subjects
        root_path = self.data_pipeline.base_data_files_path
        folders = [path for path in root_path.iterdir() if subject_num in path.name and path.is_dir()]
        paths = []
        for folder in folders:
            sub_folders = [path for path in folder.iterdir() if path.is_dir()]  # the sessions folders
            for sub_folder in sub_folders:
                files = [file for file in sub_folder.iterdir() if file.is_file() and file.suffix == '.edf']  # edf files
                paths.extend(files)
        return paths

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

    def get_my_experiments(self, experiments: list[str] or str) -> list[str]:
        """returns the experiments that are in the subject from the given list of experiments"""
        if isinstance(experiments, str):
            experiments = [experiments]

        my_experiments = [rec.experiment for rec in self.recordings if
                          any([rec.match_experiment(exp) for exp in experiments])]
        return my_experiments

    def get_dataset(self, experiments: list[str] or str) -> (np.array, np.array):
        """extract a dataset of the given experiments from the subject"""
        if isinstance(experiments, str):
            experiments = [experiments]

        datasets = [rec.get_dataset() for rec in self.recordings if
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


class Recording_Emg:
    emg_chan_order = ['EMG Ch-1', 'EMG Ch-2', 'EMG Ch-3', 'EMG Ch-4', 'EMG Ch-5', 'EMG Ch-6', 'EMG Ch-7', 'EMG Ch-8',
                      'EMG Ch-9', 'EMG Ch-10', 'EMG Ch-11', 'EMG Ch-12', 'EMG Ch-13', 'EMG Ch-14', 'EMG Ch-15',
                      'EMG Ch-16']

    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        self.files_path: list[Path] = files_path
        self.experiment: str = self.file_path_to_experiment(files_path[0])  # str, "subject_session_position" template
        self.pipeline: Data_Pipeline = pipeline  # stores all preprocessing parameters
        self.raw_edf_emg: mne.io.edf.edf.RawEDF or None = None  # emg channels only
        self.annotations: list[(float, str)] = []  # (time onset (seconds), description) pairs
        self.annotations_data: list[(np.array, str)] = []  # (EMG, acc, label) triplets
        self.segments: np.array = None  # EMG segments data stacked on dim 0
        self.labels: np.array = np.empty(0)  # labels of the segments
        self.features: np.array = np.empty(0)  # np.array, stacked on dim 0
        self.gesture_counter: dict = {}  # dict of the number of repetitions of each gesture, currently not used

    @staticmethod
    def file_path_to_experiment(file_path: Path) -> str:
        """
        extract the experiment name from the file path, currently the file name is in the form of:
        'GR_pos#_###_S#_part#_Recording_00_SD_edited.edf'
        and we want to extract the experiment in the form of: 'subject_session_position'
        """
        file_name_parts = file_path.stem.split('_')
        for name in file_name_parts:
            if name[0].upper() == 'S' and name[1].upper() != 'D':
                session = name[1:]
            elif name[0:3].upper() == 'POS':
                position = name[3:]
            elif len(name) == 3 and name.isdigit():
                subject = name
        try:
            # noinspection PyUnboundLocalVariable
            experiment = f'{subject}_{session}_{position}'
        except NameError:
            raise NameError(f'Error: could not extract experiment name from file path: {file_path}.'
                            f'pls check the file name format.')
        return experiment

    def load_file(self):
        """this function loads the files data and sets the raw_edf and annotations field.
        in the future we might insert here the handling of merging files of part1, part2, etc. to one file"""

        files_names = [str(path) for path in self.files_path]
        files_names.sort()  # part1, part2, etc. should be in order
        for file_name in files_names:
            raw_edf = mne.io.read_raw_edf(file_name, preload=True, stim_channel='auto', verbose=False)
            curr_raw_edf_emg = raw_edf.copy().pick_channels(set(self.emg_chan_order))
            curr_raw_edf_emg = curr_raw_edf_emg.reorder_channels(self.emg_chan_order)
            # concat files of the same experiment
            if self.raw_edf_emg is None:
                self.raw_edf_emg = curr_raw_edf_emg
            else:
                self.raw_edf_emg.append(curr_raw_edf_emg)
        # extract the annotations from the raw_edf, leave only relevant (and good) ones. description format:
        # '<Start\Release>_<gesture name>_<repetition number>'
        annotations = self.raw_edf_emg.annotations  # use the concatenated raw_edf_emg object to get the annotations
        annotations = [(onset, description) for onset, description in
                       zip(annotations.onset, annotations.description)
                       if 'Start' in description or 'Release' in description or 'End' in description]
        annotations = self.verify_annotations(annotations)
        self.annotations = annotations

    def verify_annotations(self, annotations: list[(float, str)]) -> list[(float, str)]:
        """
        This function verifies that the annotations are in the right order - start, stop, start, stop, etc.,
        and that each consecutive start, stop annotations are of the same gesture, where targets are in the format of:
        Start_<gesture_name>_<number> and Release_<gesture_name>_<number>
        """
        # reject unwanted annotations - keep only gesture related ones
        gesture_annotations = []
        for onset, description in annotations:
            if description == 'Recording_Emg_Acc Started' or \
                    description == 'Start Experiment' or \
                    description == 'App End Recording_Emg_Acc':
                continue
            else:
                gesture_annotations.append((onset, description))
        annotations = gesture_annotations
        # verify that the annotations are in the right order - start, stop, start, stop, etc.
        counter = {}
        verified_annotations = []
        for i, annotation in enumerate(annotations):
            if 'Start' in annotation[1]:
                start_description = annotation[1].replace('Start', '').strip('_ ')
                if 'Release' in annotations[i + 1][1] or 'End' in annotations[i + 1][1]:
                    end_description = annotations[i + 1][1].replace('Release', '').replace('End', '').strip('_ ')
                    max_gesture_duration = self.pipeline.max_gesture_duration
                    if start_description != end_description:
                        print(f'Warning: annotation mismatch of {start_description} in time: {annotation[0]}'
                              f'and {end_description} in time: {annotations[i + 1][0]},'
                              f' in the experiment: {self.experiment}')
                        continue
                    elif annotations[i + 1][0] - annotation[0] > max_gesture_duration:
                        print(f'Warning: gesture {start_description} in time {annotation[0]} in experiment '
                              f'{self.experiment} is longer than {max_gesture_duration} seconds, removing it from the '
                              f'annotations. pls check the annotations in the raw file for further details.')
                    else:
                        # add the gesture number if it doesnt exist in the label, its either
                        # todo: this is only temporary until we figure out how to handle the gesture number in the raw files
                        if start_description.split(' ')[-1].isdigit():
                            num_gest = int(start_description.split(' ')[-1])
                            counter[start_description] = num_gest if num_gest > counter.get(start_description, 0) else \
                                counter.get(start_description, 0)
                            start_description = start_description.split(' ')[0]  # remove the digit
                            end_description = end_description.split(' ')[0]  # remove the digit
                            verified_annotations.append((annotation[0], f'Start_{start_description}_{num_gest}'))
                            verified_annotations.append(
                                    (annotations[i + 1][0], f'Release_{end_description}_{num_gest}'))
                        else:
                            gest_num = counter.get(start_description, -1) + 1
                            counter[start_description] = gest_num
                            verified_annotations.append((annotation[0], f'Start_{start_description}_{gest_num}'))
                            verified_annotations.append(
                                    (annotations[i + 1][0], f'Release_{end_description}_{gest_num}'))
                else:
                    print(f'Error: annotation mismatch, no Release/End annotation for {start_description} in time: '
                          f'{annotation[0]}, in the experiment: {self.experiment}')
            else:
                continue

        # remove bad annotations - if we have the same gesture in two annotations remove the first one
        good_annotations = []
        annotations_description = [annotation[1] for annotation in verified_annotations]
        for i, annotation in enumerate(verified_annotations):
            if annotation[1] not in annotations_description[i + 1:]:
                good_annotations.append(annotation)
            else:
                print(f'Warning: annotation {annotation[1]} in time {annotation[0]} in experiment '
                      f'{self.experiment} is a duplicate, removing it from the annotations. pls check the annotations'
                      f' in the raw files for further details.')

        return good_annotations

    def preprocess_data(self) -> None:
        """
        preprocess the segments according to the data pipeline
        - load data if not loaded
        - filter the data
        - extract the annotated data and its annotation
        - segment the data (discrete or continuous)
        - normalize the data
        - extract features
        - normalize the features
        all the preprocessing steps are done in place, meaning that the data is changed in the object.
        """
        if self.raw_edf_emg is None:
            self.load_file()  # sets the raw objects and the annotations properties

        self.filter_signal()
        self.extract_annotated_data()  # list[(np.array, np.array, str)]

        if self.pipeline.segmentation_type == "discrete":
            self.segment_data_discrete()
        elif self.pipeline.segmentation_type == "continuous":
            self.segment_data_continuous()

        self.normalize_segments()
        self.extract_features()
        self.normalize_features()

    def filter_signal(self) -> None:
        # todo: check the order of the filters, decide what order to use.
        # todo: figure out if filtering the accelerometer data is necessary
        """filter the signal according to the data pipeline (currently only the EMG, acc will be added later here)"""
        raw_edf_emg = self.raw_edf_emg.notch_filter(freqs=self.pipeline.emg_notch_freq)
        raw_edf_emg = raw_edf_emg.filter(l_freq=self.pipeline.emg_low_freq, h_freq=self.pipeline.emg_high_freq,
                                         method='iir')
        self.raw_edf_emg: type(mne.io.edf.edf.RawEDF) = raw_edf_emg

    def extract_annotated_data(self) -> None:
        """extract the data of the gestures. It might be used for exploring the data, or for discrete segmentation"""
        annotations_data = []
        start_buffer = self.pipeline.annotation_delay_start  # seconds
        end_buffer = self.pipeline.annotation_delay_end  # seconds
        for i, annotation in enumerate(self.annotations):
            time, description = annotation
            if 'Release_' in description:
                continue
            elif 'Start_' in description:
                start_time = time + start_buffer
                end_time = self.annotations[i + 1][0] - end_buffer  # the next annotation is the end of the gesture
                label = description.replace('Start_', '')  # remove the 'Start_' from the label
                label_for_counter = label.strip('_0123456789')  # remove the number from the label
                self.gesture_counter[label_for_counter] = self.gesture_counter.get(label_for_counter, 0) + 1
                emg_data: type(np.arrray) = self.raw_edf_emg.get_data(tmin=start_time, tmax=end_time)
                annotations_data.append((emg_data, label))
        self.annotations_data = annotations_data

    def heatmap_visualization(self, data: np.array, num_gestures: int, num_repetitions_per_gesture: dict):
        # If the channels are not arranged in chronological order,
        # should get the location and rearrange them here.

        heatmaps = [np.reshape(segment, (4, 4)) for segment in data]  # np iterates over dim 0
        fig, axes = plt.subplots(num_gestures, max(num_repetitions_per_gesture.values()))
        idx = 0
        unique_labels = np.unique(self.labels)
        for i in range(num_gestures):
            for j in range(num_repetitions_per_gesture[unique_labels[i]]):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                im = axes[i, j].imshow(heatmaps[idx], cmap='hot')
                idx += 1
            fig.axes[i * max(num_repetitions_per_gesture.values())].set_ylabel('h', rotation=0, fontsize=18)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        # noinspection PyUnboundLocalVariable
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.tick_params(labelsize=18)
        plt.show()

    def segment_data_discrete(self) -> None:
        """
        discrete segmentation of the data according to the annotations. this is to repeat last year results.
        use float16 type to save memory
        """
        segment_length_emg = floor(self.pipeline.segment_length_sec * self.pipeline.emg_sample_rate)
        segments_emg = []
        labels = []
        step_size = floor(self.pipeline.segment_step_sec * self.pipeline.emg_sample_rate)
        for emg_data, acc_data, label in self.annotations_data:
            for i in range(0, emg_data.shape[1] - segment_length_emg, step_size):
                curr_emg_segment = emg_data[:, i:i + segment_length_emg][np.newaxis, :, :]
                segments_emg.append(curr_emg_segment)
                labels.append(label)

        self.segments = np.concatenate(segments_emg, axis=0, dtype=np.float32)
        self.labels = np.array(labels, dtype=str)  # notice that the number of the gesture is included in the labels!

    def segment_data_continuous(self) -> None:
        raise NotImplementedError

    def normalize_segments(self) -> None:
        self.segments = self.norm_me(self.segments, self.pipeline.emg_norm)

    def normalize_features(self) -> np.array:
        self.features = self.norm_me(self.features, self.pipeline.features_norm)

    @staticmethod
    def norm_me(data: np.array, norm_type: str) -> np.array:
        axis = tuple(range(1, data.ndim))
        if norm_type == 'zscore':
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            data = (data - mean) / std
        elif norm_type == '01':
            min_ = np.min(data, axis=axis, keepdims=True)
            max_ = np.max(data, axis=axis, keepdims=True)
            data = (data - min_) / (max_ - min_)
        elif norm_type == '-11':
            min_ = np.min(data, axis=axis, keepdims=True)
            max_ = np.max(data, axis=axis, keepdims=True)
            data = 2 * (data - min_) / (max_ - min_) - 1
        elif 'quantile' in norm_type:
            quantiles = norm_type.split('_')[1].split('-')
            quantiles = [float(q) for q in quantiles]
            low_quantile = np.quantile(data, quantiles[0], axis=axis, keepdims=True)
            high_quantile = np.quantile(data, quantiles[1], axis=axis, keepdims=True)
            data = (data - low_quantile) / (high_quantile - low_quantile)
        elif norm_type == 'max':
            max_ = np.max(data, axis=axis, keepdims=True)
            data = data / max_
        elif norm_type == 'none':
            pass
        else:
            raise ValueError('Invalid normalization method for EMG data')
        return data

    def extract_features(self) -> None:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        segments = (self.segments, None, None)  # (emg, acc, gyro)
        features = feature_extractor.extract_features(segments, **self.pipeline.features_extraction_params)
        self.features = features

    def get_dataset(self) -> (np.array, np.array):
        """extract a dataset of the given recording file"""
        if self.features.size == 0:  # empty array, no features were extracted
            self.preprocess_data()

        labels = self.labels
        labels = [f'{self.experiment}_{label}' for label in labels]  # add the experiment name to the labels

        return self.features, labels

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


class Recording_Emg_Acc(Recording_Emg):
    acc_chan_order_1 = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    acc_chan_order_2 = ['ACC-X', 'ACC-Y', 'ACC-Z']

    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        self.files_path: list[Path] = files_path
        self.experiment: str = self.file_path_to_experiment(files_path[0])  # str, "subject_session_position" template
        self.pipeline: Data_Pipeline = pipeline  # stores all preprocessing parameters
        self.raw_edf_emg: mne.io.edf.edf.RawEDF or None = None  # emg channels only
        self.raw_edf_acc: mne.io.edf.edf.RawEDF or None = None  # accelerometer channels only
        self.annotations: list[(float, str)] = []  # (time onset (seconds), description) pairs
        self.annotations_data: list[(np.array, np.array, str)] = []  # (EMG, acc, label) triplets
        self.segments: (np.array, np.array) = ()  # (EMG, acc) segments data stacked on dim 0
        self.labels: np.array = np.empty(0)  # labels of the segments
        self.features: np.array = np.empty(0)  # np.array, stacked on dim 0
        self.gesture_counter: dict = {}  # dict of the number of repetitions of each gesture, currently not used

    def load_file(self):
        """this function loads the files data and sets the raw_edf and annotations field.
        in the future we might insert here the handling of merging files of part1, part2, etc. to one file"""

        files_names = [str(path) for path in self.files_path]
        files_names.sort()  # part1, part2, etc. should be in order
        for file_name in files_names:
            raw_edf = mne.io.read_raw_edf(file_name, preload=True, stim_channel='auto', verbose=False)
            # get the raw edf channels names
            channels_names = raw_edf.ch_names
            if 'Accelerometer_X' in channels_names:
                acc_names = self.acc_chan_order_1
            elif 'ACC-X' in channels_names:
                acc_names = self.acc_chan_order_2
            else:
                raise NameError(f'Error: could not find accelerometer channels in file: {file_name}.'
                                f'pls check the file name format.')

            curr_raw_edf_emg = raw_edf.copy().drop_channels(set(acc_names))
            curr_raw_edf_emg = curr_raw_edf_emg.reorder_channels(self.emg_chan_order)
            curr_raw_edf_acc = raw_edf.copy().pick_channels(set(acc_names))
            curr_raw_edf_acc = curr_raw_edf_acc.reorder_channels(acc_names)

            # concat files of the same experiment
            if self.raw_edf_acc is None:
                self.raw_edf_acc = curr_raw_edf_acc
                self.raw_edf_emg = curr_raw_edf_emg
            else:
                self.raw_edf_acc.append(curr_raw_edf_acc)
                self.raw_edf_emg.append(curr_raw_edf_emg)
        # extract the annotations from the raw_edf, leave only relevant (and good) ones. description format:
        # '<Start\Release>_<gesture name>_<repetition number>'
        annotations = self.raw_edf_emg.annotations  # use the concatenated raw_edf_emg object to get the annotations
        annotations = [(onset, description) for onset, description in
                       zip(annotations.onset, annotations.description)
                       if 'Start' in description or 'Release' in description or 'End' in description]
        annotations = self.verify_annotations(annotations)
        self.annotations = annotations

    def filter_signal(self) -> None:
        # todo: check the order of the filters, decide what order to use.
        # todo: figure out if filtering the accelerometer data is necessary
        """filter the signal according to the data pipeline (currently only the EMG, acc will be added later here)"""
        super().filter_signal()
        self.raw_edf_acc: type(mne.io.edf.edf.RawEDF) = self.raw_edf_acc

    def extract_annotated_data(self) -> None:
        """extract the data of the gestures. It might be used for exploring the data, or for discrete segmentation"""
        annotations_data = []
        start_buffer = self.pipeline.annotation_delay_start  # seconds
        end_buffer = self.pipeline.annotation_delay_end  # seconds
        for i, annotation in enumerate(self.annotations):
            time, description = annotation
            if 'Release_' in description:
                continue
            elif 'Start_' in description:
                start_time = time + start_buffer
                end_time = self.annotations[i + 1][0] - end_buffer  # the next annotation is the end of the gesture
                label = description.replace('Start_', '')  # remove the 'Start_' from the label
                label_for_counter = label.strip('_0123456789')  # remove the number from the label
                self.gesture_counter[label_for_counter] = self.gesture_counter.get(label_for_counter, 0) + 1
                emg_data: type(np.arrray) = self.raw_edf_emg.get_data(tmin=start_time, tmax=end_time)
                acc_data: type(np.array) = self.raw_edf_acc.get_data(tmin=start_time, tmax=end_time)
                annotations_data.append((emg_data, acc_data, label))
        self.annotations_data = annotations_data

    def segment_data_discrete(self) -> None:
        """
        discrete segmentation of the data according to the annotations. this is to repeat last year results.
        use float16 type to save memory
        """
        segment_length_emg = floor(self.pipeline.segment_length_sec * self.pipeline.emg_sample_rate)
        segment_length_acc = floor(self.pipeline.segment_length_sec * self.pipeline.acc_sample_rate)

        segments_emg = []
        segments_acc = []
        labels = []
        step_size = floor(self.pipeline.segment_step_sec * self.pipeline.emg_sample_rate)
        for emg_data, acc_data, label in self.annotations_data:
            # emg segmentation and labels creation
            for i in range(0, emg_data.shape[1] - segment_length_emg, step_size):
                curr_emg_segment = emg_data[:, i:i + segment_length_emg][np.newaxis, :, :]
                segments_emg.append(curr_emg_segment)
                labels.append(label)
            # acc segmentation
            for i in range(0, acc_data.shape[1] - segment_length_acc, step_size):
                curr_acc_segment = acc_data[:, i:i + segment_length_acc][np.newaxis, :, :]
                segments_acc.append(curr_acc_segment)

        segments_emg = np.concatenate(segments_emg, axis=0, dtype=np.float32)
        segments_acc = np.concatenate(segments_acc, axis=0, dtype=np.float32)
        labels = np.array(labels, dtype=str)

        self.segments = (segments_emg, segments_acc)
        self.labels = labels  # notice that the number of the gesture is included in the labels!

    def segment_data_continuous(self) -> None:
        raise NotImplementedError

    def normalize_segments(self) -> None:
        acc_segments = self.segments[1]
        emg_segments = self.segments[0]
        acc_segments = self.norm_me(acc_segments, self.pipeline.acc_norm)
        emg_segments = self.norm_me(emg_segments, self.pipeline.emg_norm)
        self.segments = (emg_segments, acc_segments)

    def extract_features(self) -> None:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        segments = (self.segments[0], self.segments[1], None)  # emg, acc, gyro
        features = feature_extractor.extract_features(segments, **self.pipeline.features_extraction_params)
        self.features = features


class Real_Time_Recording:
    def __init__(self, data_streamer: Data, pipeline: Data_Pipeline):
        self.data_streamer = data_streamer
        self.pipeline = pipeline
        self.latest_segment = None  # currenly includes only the emg data
        self.latest_features = None

    def get_feats_for_prediction(self) -> np.array:
        """
        The main function of the class, this should be the only function called from outside the class.
        This function is responsible for fetching the latest segment from the data streamer, preprocess it and return
        the features ready for prediction.

        Note: currently this class only takes care of the emg data since the streamer class doesn't support acc and gyro data yet.
            once the streamer class will support acc and gyro data, this class will be updated to support them as well.
        """
        self._fetch_latest_segment()
        self._preprocess_latest_segment()
        return self.latest_features

    def _fetch_latest_segment(self) -> None:
        """fetch the latest segment from the data streamer"""
        self.latest_segment = self.data_streamer.exg_data[
                              -self.data_streamer.fs_exg * self.pipeline.segment_length_sec:, :]

    def _preprocess_latest_segment(self) -> None:
        """
        preprocess the segment according to the data pipeline
        - filter the data
        - normalize the data
        - extract features
        - normalize the features
        all the preprocessing steps are done in place, meaning that the data is changed in the object.
        """
        self._filter_segment()
        self._normalize_segment()
        self._extract_features()
        self._normalize_features()

    def _filter_segment(self) -> None:
        # todo: check the order of the filters, decide what order to use.
        # todo: figure out if filtering the accelerometer data is necessary
        """filter the signal according to the data pipeline (currently only the EMG, acc will be added later here)"""
        fs = self.data_streamer.fs_exg
        l_freq = self.pipeline.emg_low_freq
        h_freq = self.pipeline.emg_high_freq
        notch = self.pipeline.emg_notch_freq
        if l_freq or h_freq:
            self.latest_segment = mne.filter.filter_data(self.latest_segment, sfreq=fs, l_freq=l_freq, h_freq=h_freq,
                                                         method='iir', verbose=False)
        if notch:
            self.latest_segment = mne.filter.notch_filter(self.latest_segment, Fs=fs, freqs=notch, method='iir',
                                                          verbose=False)

    def _normalize_segment(self) -> None:
        self.latest_segment = self._norm_me(self.latest_segment, self.pipeline.emg_norm)

    def _extract_features(self) -> None:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        segment = (self.latest_segment, None, None)
        self.latest_features = feature_extractor.extract_features(segment, **self.pipeline.features_extraction_params)

    def _normalize_features(self) -> None:
        self.latest_features = self._norm_me(self.latest_features, self.pipeline.features_norm)

    @staticmethod
    def _norm_me(data: np.array, norm_type: str) -> np.array:
        if norm_type == 'zscore':
            mean = np.mean(data, keepdims=True)
            std = np.std(data, keepdims=True)
            data = (data - mean) / std
        elif norm_type == '01':
            min_ = np.min(data, keepdims=True)
            max_ = np.max(data, keepdims=True)
            data = (data - min_) / (max_ - min_)
        elif norm_type == '-11':
            min_ = np.min(data, keepdims=True)
            max_ = np.max(data, keepdims=True)
            data = 2 * (data - min_) / (max_ - min_) - 1
        elif 'quantile' in norm_type:
            quantiles = norm_type.split('_')[1].split('-')
            quantiles = [float(q) for q in quantiles]
            low_quantile = np.quantile(data, quantiles[0], keepdims=True)
            high_quantile = np.quantile(data, quantiles[1], keepdims=True)
            data = (data - low_quantile) / (high_quantile - low_quantile)
        elif norm_type == 'max':
            max_ = np.max(data, keepdims=True)
            data = data / max_
        elif norm_type == 'none':
            pass
        else:
            raise ValueError('Invalid normalization method for EMG data')
        return data


######################################################
# Feature Extractors
######################################################
class Feature_Extractor(ABC):
    @abstractmethod
    def extract_features(self, segments: (np.array, np.array, np.array), **kwargs) -> np.array:
        """
        This function is responsible for extracting features from the segments.
        inputs:
            segments: tuple of (emg_segments, acc_segments, gyro_segments), emg_segments is a 3d array of shape
            (num_segments, num_channels, segment_length) where num_channels is constant 16, acc_segments is a 3d array of
                      shape (num_segments, num_channels, segment_length) where num_channels is constant 3 (x, y, z - in
                      that order), gyro data is tbd.
            kwargs: not used
        """
        pass


class RMS_Feature_Extractor(Feature_Extractor):
    def extract_features(self, segments: (np.array, np.array, np.array), output_shape: tuple = (1, 4, 4)) -> np.array:
        """
        extract the RMS of each emg channel and reshape into a 4x4 array.
        input:
            segments: tuple of (emg_segments, acc_segments), emg_segments is a 3d array of shape (num_segments,
            num_channels, segment_length) where num_channels is constant 16, acc_segments is a 3d array of shape
            (num_segments, num_channels, segment_length) where num_channels is constant 3 (x, y, z - not in that order)
            kwargs: not used
            output_shape: the shape of the output array of each segment, not including the number of segments which will be set as the first
            dimension (dim 0). the sum of the elements in the tuple must be equal to the number of emg channels - 16.
        output:
            features: array of shape (num_segments, 4, 4)
        """
        emg_data, _, _ = segments

        if emg_data.ndim == 3:  # offline analysis
            axis = 2
            n_windows = emg_data.shape[0]
        else:  # online analysis
            axis = 1
            emg_data = emg_data.T if emg_data.shape[0] > emg_data.shape[1] else emg_data
            n_windows = 1

        rms = np.atleast_2d(np.sqrt(np.mean(np.square(emg_data), axis=axis)))
        features = rms.reshape(n_windows, *output_shape)  # reshape to the desired output shape
        return features


"""Builder"""
extractors = {"RMS": RMS_Feature_Extractor()}


def build_feature_extractor(method: str) -> Feature_Extractor:
    try:
        extractor = extractors[method]
        return extractor
    except KeyError:
        raise ValueError(f"Invalid method name: {method}")
