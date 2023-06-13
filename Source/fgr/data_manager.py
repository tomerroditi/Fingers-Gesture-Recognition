"""This module contains the data manager class and its helper classes. It is used to construct datasets from raw data
files using the data pipeline. The public API of the module is the Data_Manager class."""
import collections
import numpy as np
import matplotlib.pyplot as plt
import mne

from hmmlearn import hmm
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
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
#       2. we process the data once, and then we can use it as many emg_times as we want.
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

        if load_data:
            [rec._load_raw_data_and_annotations() for rec in  # noqa
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


class Base_Recording:

    def __init__(self, pipeline: Data_Pipeline):
        self.pipeline = pipeline

    @staticmethod
    def _filter_data(data: np.ndarray, fs: float, notch: float, low_freq: float, high_freq: float,
                     buff_len: int = 0) -> np.ndarray:
        """filter the data according to the pipeline

        Parameters
        ----------
        data : np.ndarray
            the data to filter, shape: (n_segments, n_channels, n_samples)

        Returns
        -------
        np.ndarray
            the filtered data, shape: (n_gestures, n_channels, n_samples - filter_buffer * sample_rate)
        """
        # notch filter design
        Q = 30  # Quality factor
        w0 = notch / (fs / 2)  # Normalized frequency
        b_notch, a_notch = iirnotch(w0, Q)

        # band pass filter design
        low_band = low_freq / (fs / 2)
        high_band = high_freq / (fs / 2)
        # create bandpass filter for EMG
        sos = butter(4, [low_band, high_band], btype='bandpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        data = sosfiltfilt(sos, data, axis=2, padtype='even')
        data = filtfilt(b_notch, a_notch, data, axis=2, padtype='even')

        if buff_len > 0:
            data = data[:, :, buff_len:]
        return data

    @staticmethod
    def normalize_data(data: np.array, norm_type: str) -> np.array:
        """
        normalize the data according to the pipeline

        Parameters
        ----------
        data : np.ndarray
            the data to normalize, shape: (n_segments, n_channels, n_samples)
        norm_type : str
            the type of normalization to apply, one of: 'max', 'zscore', '01', '-11', 'quantileX' where X is the
            quantile to use for normalization
        """
        if norm_type == 'none':  # no normalization, fast return for online performance
            return data

        axis = (1, 2)  # normalize over the channels and samples

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
        else:
            raise ValueError('Invalid normalization method for EMG data')
        return data

    def extract_features(self, emg: np.ndarray, acc: np.ndarray = None, gyro: np.ndarray = None) -> np.ndarray:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        segments = (emg, acc, gyro)  # (emg, acc, gyro)
        features = feature_extractor.extract_features(segments, **self.pipeline.features_extraction_params)
        return features


class Recording_Emg(Base_Recording):
    emg_chan_order_1 = ['EMG Ch-1', 'EMG Ch-2', 'EMG Ch-3', 'EMG Ch-4', 'EMG Ch-5', 'EMG Ch-6', 'EMG Ch-7', 'EMG Ch-8',
                        'EMG Ch-9', 'EMG Ch-10', 'EMG Ch-11', 'EMG Ch-12', 'EMG Ch-13', 'EMG Ch-14', 'EMG Ch-15',
                        'EMG Ch-16']
    emg_chan_order_2 = ['Channel 0', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6',
                        'Channel 7', 'Channel 8', 'Channel 9', 'Channel 10', 'Channel 11', 'Channel 12', 'Channel 13',
                        'Channel 14', 'Channel 15']

    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        self.files_path: list[Path] = files_path
        self.experiment: str = self.file_path_to_experiment(files_path[0])  # str, "subject_session_position" template
        self.pipeline: Data_Pipeline = pipeline  # stores all preprocessing parameters
        self.emg_data: np.ndarray or None = None  # emg data, shape: (n_channels, n_samples)
        self.emg_times: np.ndarray or None = None  # time stamps of the emg data, shape: (n_samples,)
        self.annotations: list[(float, str)] or None = None  # (time onset (seconds), description) pairs
        self.segments: np.ndarray or None = None  # EMG segments, shape: (n_segments, n_channels, n_samples)
        self.labels: np.ndarray or None = None  # labels of the segments, shape: (n_segments,)
        self.features: np.ndarray or None = None  # features of the segments, shape: (n_segments, *features_dims)

    @staticmethod
    def file_path_to_experiment(file_path: Path) -> str:
        """
        extract the experiment name from the file path, currently the file name is in the form of:
        'GR_pos#_###_S#_part#_Recording_00_SD_edited.edf'
        or
        'GR_pos#_###_S#_part#_rep#_Recording_00_SD_edited.edf'
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
            elif 'rep' in name:
                repetition = name.strip('rep')[0]
        try:

            if 'repetition' in locals():  # online training sessions
                # noinspection PyUnboundLocalVariable
                experiment = f'{subject}_{session}_{position}_{repetition}'
            else:
                # noinspection PyUnboundLocalVariable
                experiment = f'{subject}_{session}_{position}'
        except NameError:
            raise NameError(f'Error: could not extract experiment name from file path: {file_path}.'
                            f'pls check the file name format.')
        return experiment

    def _load_raw_data_and_annotations(self, return_raw_edf: bool = False) -> \
            (np.ndarray, np.ndarray, list[(float, str)]) or (np.ndarray, np.ndarray, list[(float, str)], mne.io.Raw):
        """this function loads the files data and sets the raw_edf and annotations field.
        in the future we might insert here the handling of merging files of part1, part2, etc. to one file"""
        raw_edf = None
        files_names = [str(path) for path in self.files_path]
        files_names.sort()  # part1, part2, etc. should be in order
        for file_name in files_names:
            curr_raw_edf = mne.io.read_raw_edf(file_name, preload=True, stim_channel='auto', verbose=False)
            if raw_edf is None:
                raw_edf = curr_raw_edf
            else:
                raw_edf.append(curr_raw_edf)

        channels_names = raw_edf.ch_names
        if 'EMG Ch-1' in channels_names:
            names = self.emg_chan_order_1
        elif 'Channel 0' in channels_names:
            names = self.emg_chan_order_2
        else:
            raise ValueError('no matching channels where found, pls check the recording channels names!')

        raw_edf_emg = raw_edf.copy().pick_channels(set(names))
        raw_edf_emg = raw_edf_emg.reorder_channels(names)

        # extract raw data
        emg_data, times = raw_edf_emg.get_data(units='uV', return_times=True)  # emg data - no scaling!!!
        # extract annotations
        annotations = self._get_verified_annotations(raw_edf_emg.annotations)

        if return_raw_edf:
            return emg_data, times, annotations, raw_edf
        else:
            return emg_data, times, annotations

    def _get_verified_annotations(self, annotations: mne.Annotations) -> list[(float, str)]:
        """
        This function verifies that the annotations are in the right order - start, stop, start, stop, etc.,
        and that each consecutive start, stop annotations are of the same gesture, where targets are in the format of:
        Start_<gesture_name>_<number> and Release_<gesture_name>_<number>
        """
        annotations = [(onset, description) for onset, description in zip(annotations.onset, annotations.description)
                       if 'Start' in description or 'Release' in description or 'End' in description]
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
                start_description = annotation[1].replace('Start', '').strip('_ ').replace('_', ' ')
                if 'Release' in annotations[i + 1][1] or 'End' in annotations[i + 1][1]:
                    end_description = annotations[i + 1][1].replace('Release', '').replace('End', '').strip('_ ').\
                        replace('_', ' ')
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
                        # add the gesture number if it doesn't exist in the label, it is either the last word or the
                        # last
                        # word after '_' (offline and online formats)
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
        if self.emg_data is None:
            self.emg_data, self.emg_times, self.annotations = self._load_raw_data_and_annotations()

        fs = self.pipeline.emg_sample_rate
        buff_len = fs * self.pipeline.emg_buff_dur
        if self.pipeline.segmentation_type == "discrete":
            self.segments, self.labels = self.segment_data_discrete(self.emg_data, self.emg_times, fs,
                                                                    buff_len=buff_len)
        elif self.pipeline.segmentation_type == "continuous":
            raise NotImplementedError("continuous segmentation is not implemented yet")
        else:
            raise ValueError("invalid segmentation type")

        self.segments = self._filter_data(self.segments, self.pipeline.emg_sample_rate, self.pipeline.emg_notch_freq,
                                          self.pipeline.emg_low_freq, self.pipeline.emg_high_freq, buff_len)

        self.segments = self.normalize_data(self.segments, self.pipeline.emg_norm)
        self.features = self.extract_features(self.segments)
        self.features = self.normalize_data(self.features, self.pipeline.features_norm)

    def segment_data_discrete(self, data: np.ndarray, times: np.ndarray, fs: float, buff_len: int = 0) ->\
            (np.ndarray, np.ndarray):
        """
        discrete segmentation of the data according to the annotations.

        Parameters
        ----------
        data: np.ndarray
            the data to segment, should be of shape (n_channels, n_samples)
        fs: float
            the sampling frequency of the data
        buff_len: int
            the length of the buffer to add to the segments, in samples
        """
        seg_dur = self.pipeline.segment_length_sec
        step_dur = self.pipeline.segment_step_sec
        seg_len = floor(seg_dur * fs)
        step_size = floor(step_dur * fs)
        start_delay_len = floor(self.pipeline.annotation_delay_start * fs)
        end_delay_len = floor(self.pipeline.annotation_delay_end * fs)

        segments = []
        labels = []
        labels_vector = self._get_time_labels_vector(times)
        for i in range(max(buff_len, start_delay_len), data.shape[1] - seg_len - end_delay_len, step_size):
            if not np.all(labels_vector[i:i + seg_len] != 'Idle') or \
               not np.all(labels_vector[i - start_delay_len: i] != 'Idle') or \
               not np.all(labels_vector[i + seg_len: i + seg_len + end_delay_len] != 'Idle'):
                continue
            segments.append(data[:, i - buff_len: i + seg_len])
            labels.append(labels_vector[i])  # notice that the number of the gesture is included in the labels!

        return np.stack(segments, axis=0), np.array(labels, dtype=str)

    def segment_data_continuous(self) -> None:
        raise NotImplementedError

    def _get_time_labels_vector(self, times: np.array) -> np.ndarray:
        """This function creates a vector of labels for each time stamp to be used in the segmentation process."""
        if self.annotations is None:
            raise ValueError('annotations are not loaded, please load the annotations first with the '
                             '"_load_raw_data_and_annotations" function.')
        # create a vector of labels for each time stamp with 'idle' as the default label
        time_labels = np.full(times.shape, fill_value='Idle', dtype='<U30')
        for i in range(0, len(self.annotations), 2):
            desc = self.annotations[i][1].replace('Start_', '')
            start_time = self.annotations[i][0]
            end_time = self.annotations[i + 1][0]
            time_labels[np.logical_and(times >= start_time, times <= end_time)] = desc
        return time_labels

    def get_dataset(self) -> (np.array, np.array):
        """extract a dataset of the given recording file"""
        if self.features is None:
            self.preprocess_data()
        labels = np.char.add(f'{self.experiment}_', self.labels)  # add the experiment name to the labels
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

    def heatmap_visualization(self, data: np.array, num_gestures: int, num_repetitions_per_gesture: dict):
        raise NotImplementedError

    def get_annotated_data(self, data: np.ndarray) -> list[(np.ndarray, str)]:
        """
        extract the data that is annotated and its annotation from the raw data. the annotations are not guaranteed to
        be all the same length so we cant store the data in a single matrix. instead we store it in a list of tuples
        where each tuple is a segment of the data and its annotation.
        """
        data_segments = []
        labels_array = []
        start_buffer = self.pipeline.annotation_delay_start  # seconds
        end_buffer = self.pipeline.annotation_delay_end  # seconds
        for i, annotation in enumerate(self.annotations):
            time, description = annotation
            if 'Release_' in description:
                continue
            elif 'Start_' in description:
                start_time = time + start_buffer
                end_time = self.annotations[i + 1][0] - end_buffer  # the next annotation is the end of the gesture
                labels_array.append(description.replace('Start_', ''))
                data_segments.append(data[:, np.logical_and(self.emg_times >= start_time, self.emg_times <= end_time)])
            else:
                raise ValueError(f'annotation {description} is not valid, it should contain either Start or Release')
        return zip(data_segments, labels_array)


class Recording_Emg_Acc(Recording_Emg):
    acc_chan_order_1 = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    acc_chan_order_2 = ['ACC-X', 'ACC-Y', 'ACC-Z']
    acc_chan_3 = ['Channel 16', 'Channel 17', 'Channel 18']

    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        super().__init__(files_path, pipeline)
        self.acc_data: np.ndarray or None = None  # acc data, shape: (n_channels, n_samples)
        self.acc_times: np.ndarray or None = None  # acc times, shape: (n_samples,)
        self.segments: (np.ndarray, np.ndarray) or None = None  # (emg, acc), shapes: (n_segments, n_channels, n_samples)

    def preprocess_data(self) -> None:
        if self.emg_data is None:
            self.emg_data, self.emg_times, self.acc_data, self.acc_times, self.annotations = \
                self._load_raw_data_and_annotations()

        acc_fs = self.pipeline.acc_sample_rate
        emg_fs = self.pipeline.emg_sample_rate
        emg_buff_len = emg_fs * self.pipeline.emg_buff_dur
        if self.pipeline.segmentation_type == "discrete":
            emg_segments, self.labels = self.segment_data_discrete(self.emg_data, self.emg_times, emg_fs,
                                                                   buff_len=emg_buff_len)
            acc_segments, _ = self.segment_data_discrete(self.acc_data, self.acc_times, acc_fs)
        elif self.pipeline.segmentation_type == "continuous":
            raise NotImplementedError("continuous segmentation is not implemented yet")
        else:
            raise ValueError("invalid segmentation type")

        emg_segments = self._filter_data(emg_segments, self.pipeline.emg_sample_rate, self.pipeline.emg_notch_freq,
                                         self.pipeline.emg_low_freq, self.pipeline.emg_high_freq, emg_buff_len)
        emg_segments = self.normalize_data(emg_segments, self.pipeline.emg_norm)

        self.segments = (emg_segments, acc_segments)
        self.features = self.extract_features(emg_segments, acc=acc_segments)
        self.features = self.normalize_data(self.features, self.pipeline.features_norm)

    def _load_raw_data_and_annotations(self, return_raw_edf: bool = False) -> \
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[(float, str)]) or \
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[(float, str)], mne.io.Raw):
        """
        load the raw data and annotations from the recording file

        Parameters
        ----------
        return_raw_edf: bool
            whether to return the raw edf object or not

        Returns
        -------
        emg_data: np.ndarray
            emg data, shape: (n_channels, n_samples)
        emg_time: np.ndarray
            emg times, shape: (n_samples,)
        acc_data: np.ndarray
            acc data, shape: (n_channels, n_samples)
        acc_time: np.ndarray
            acc times, shape: (n_samples,)
        annotations: list[(float, str)]
            list of tuples where each tuple is an annotation and its time
        raw_edf: mne.io.Raw
            raw edf object, only returned if return_raw_edf is True
        """
        # get emg data and raw edf
        emg_data, emg_time, annotations, raw_edf = super()._load_raw_data_and_annotations(return_raw_edf=True)

        # extract acc data
        channels_names = raw_edf.ch_names
        if 'Accelerometer_X' in channels_names:
            acc_names = self.acc_chan_order_1
        elif 'ACC-X' in channels_names:
            acc_names = self.acc_chan_order_2
        elif 'Channel 16' in channels_names:
            acc_names = self.acc_chan_3
        else:
            raise NameError(f'no matching channels where found, pls check the recording channels names!')

        raw_edf_acc = raw_edf.copy().pick_channels(set(acc_names))
        raw_edf_acc = raw_edf_acc.reorder_channels(acc_names)
        # todo: check if we need to set a unit here to prevent scaling the data
        acc_data, acc_time = raw_edf_acc.get_data(return_times=True, units='uV')

        if return_raw_edf:
            return emg_data, emg_time, acc_data, acc_time, annotations, raw_edf
        else:
            return emg_data, emg_time, acc_data, acc_time, annotations


class Real_Time_Recording(Base_Recording):
    def __init__(self, data_streamer: Data, pipeline: Data_Pipeline):
        self.data_streamer = data_streamer
        self.pipeline = pipeline

        seg_dur = self.pipeline.segment_length_sec
        fs = self.pipeline.emg_sample_rate
        buff_dur = self.pipeline.emg_buff_dur
        self.seg_len = floor(seg_dur * fs)
        self.buff_len = floor(buff_dur * fs)
        self.total_len = self.seg_len + self.buff_len

    def get_feats_for_prediction(self) -> np.ndarray:
        """
        The main function of the class, this should be the only function called from outside the class.
        This function is responsible for fetching the latest segment from the data streamer, preprocess it and return
        the features ready for prediction.

        Returns
        -------
        features: np.ndarray
            the features ready for prediction, the shape is dependent on the feature extraction function that is set
            in the pipeline. shape: (1, *features_shape)

        Note: currently this class only takes care of the emg data since the streamer class doesn't support acc and gyro
              data yet. once the streamer class will support acc and gyro data, this class will be updated to support
              them as well.
        """
        segment = self._fetch_latest_segment()
        # self._plot_segment(segment[0])
        if self.data_streamer.fs_exg != self.pipeline.emg_sample_rate:
            raise ValueError("the sampling rate of the data streamer and the pipeline must be the same")

        segment = self._filter_data(segment, self.pipeline.emg_sample_rate, self.pipeline.emg_notch_freq,
                                    self.pipeline.emg_low_freq, self.pipeline.emg_high_freq, self.buff_len)
        segment = self.normalize_data(segment, self.pipeline.emg_norm)
        # self._plot_segment(segment[0])
        features = self.extract_features(segment)
        features = self.normalize_data(features, self.pipeline.features_norm)
        return features

    def _fetch_latest_segment(self) -> np.ndarray:
        """
        fetch the latest segment from the data streamer.

        Returns
        -------
        latest_segment: np.ndarray
            the latest segment fetched from the data streamer, shape: (1, n_channels, n_samples)
        """
        # transpose to match the shape of the preprocessing functions
        # add a dimension to the data to match the input shape of the preprocessing functions
        return self.data_streamer.exg_data[- self.total_len:, :].T[np.newaxis, :, :]

    @staticmethod
    def _plot_segment(segment: np.ndarray) -> None:
        plt.figure()
        for i in range(16):
            plt.plot(segment[:, i], label=f'channel {i + 1}')
        plt.legend()
        plt.ylim(-200, 200)
        plt.show()


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
                      that order), gyro data is a 3d array of shape (num_segments, num_channels, segment_length) where
                        num_channels is constant 3 (x, y, z - in that order).
                        note that acc and gyro data might be unavailable in some recordings, in that case their value
                        will be None.
            kwargs: dict
                any additional arguments that the feature extractor might need.
        """
        pass


class RMS_Feature_Extractor(Feature_Extractor):
    def extract_features(self, segments: (np.array, np.array, np.array), output_shape: tuple = (1, 4, 4)) -> np.array:
        """
        extract the RMS of each emg channel and reshape into a 4x4 array.

        Parameters
        ----------
        segments: tuple of (emg_segments, acc_segments, gyro_segments). for more info check the docstring of the
                    extract_features method in Feature_Extractor class.
        output_shape: tuple
            the desired output shape of the features, default: (1, 4, 4)

        Returns
        -------
        features: np.array
            the extracted features, shape: (num_segments, 4, 4)
        """
        emg_data, _, _ = segments
        rms = np.atleast_2d(np.sqrt(np.mean(np.square(emg_data), axis=2)))
        features = rms.reshape(emg_data.shape[0], *output_shape)  # reshape to the desired output shape
        return features


"""Builder"""
extractors = {"RMS": RMS_Feature_Extractor()}


def build_feature_extractor(method: str) -> Feature_Extractor:
    try:
        extractor = extractors[method]
        return extractor
    except KeyError:
        raise ValueError(f"Invalid method name: {method}")
