import numpy as np
import matplotlib.pyplot as plt
from .pipelines import Data_Pipeline
from pathlib import Path
from math import floor
from .feature_extractors import build_feature_extractor
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
# todo: start working with the raw files and not the edited files, we need the gesture number for the labels and they
#       removed it!


class Recording:
    emg_chan_order = ['EMG Ch-1', 'EMG Ch-2', 'EMG Ch-3', 'EMG Ch-4', 'EMG Ch-5', 'EMG Ch-6', 'EMG Ch-7', 'EMG Ch-8',
                      'EMG Ch-9', 'EMG Ch-10', 'EMG Ch-11', 'EMG Ch-12', 'EMG Ch-13', 'EMG Ch-14', 'EMG Ch-15',
                      'EMG Ch-16']
    acc_chan_order = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']

    def __init__(self, files_path: list[Path], pipeline: Data_Pipeline):
        self.files_path: list[Path] = files_path
        self.experiment: str = self.file_path_to_experiment(files_path[0])  # str, "subject_session_position" template
        self.pipeline: Data_Pipeline = pipeline  # stores all preprocessing parameters
        self.raw_edf_emg: mne.io.edf.edf.RawEDF = None  # emg channels only
        self.raw_edf_acc: mne.io.edf.edf.RawEDF = None  # accelerometer channels only
        self.annotations: list[(float, str)] = []  # (time onset (seconds), description) pairs
        self.annotations_data: list[(np.array, np.array, str)] = []  # (EMG, acc, label) triplets
        self.segments: (np.array, np.array) = ()  # (EMG, acc) segments data stacked on dim 0
        self.labels: np.array = np.empty(0)  # labels of the segments
        self.features: np.array = np.empty(0)  # np.array, stacked on dim 0
        self.synthetic_features: np.array = np.empty(0)  # np.array, stacked on dim 0
        self.synthetic_labels: np.array = np.empty(0)  # np.array, stacked on dim 0
        self.gesture_counter: dict = {}  # dict of the number of repetitions of each gesture

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
            elif len(name) == 3:
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
        # TODO: we need to add raw files fixing as well (deleting the bad data, removing bad annotations, etc.)

        files_names = [str(path) for path in self.files_path]
        files_names.sort()  # part1, part2, etc. should be in order
        for file_name in files_names:
            raw_edf = mne.io.read_raw_edf(file_name, preload = True, stim_channel = 'auto', verbose = False)
            curr_raw_edf_acc = raw_edf.copy().pick_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
            curr_raw_edf_emg = raw_edf.copy().drop_channels({"Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"})
            # rearrange the channels to create a uniform order between all recordings
            curr_raw_edf_acc = curr_raw_edf_acc.reorder_channels(self.acc_chan_order)
            curr_raw_edf_emg = curr_raw_edf_emg.reorder_channels(self.emg_chan_order)
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
                            if 'Start_' in description or 'Release_' in description]
        annotations = self.verify_annotations(annotations)
        self.annotations = annotations

    def verify_annotations(self, annotations: list[(float, str)]) -> list[(float, str)]:
        """
        This function verifies that the annotations are in the right order - start, stop, start, stop, etc.,
        and that each consecutive start, stop annotations are of the same gesture, where targets are in the format of:
        Start_<gesture_name>_<number> and Release_<gesture_name>_<number>
        """
        counter = {}
        verified_annotations = []
        for i, annotation in enumerate(annotations):
            if 'Start_' in annotation[1]:
                start_description = annotation[1].replace('Start_', '').strip()
                if 'Release_' in annotations[i + 1][1]:
                    end_description = annotations[i + 1][1].replace('Release_', '').strip()
                    max_gesture_duration = self.pipeline.max_gesture_duration
                    if start_description != end_description:
                        print(f'Error: annotation mismatch of {start_description} in time: {annotation[0]}'
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
                            verified_annotations.append((annotations[i + 1][0], f'Release_{end_description}_{num_gest}'))
                        else:
                            gest_num = counter.get(start_description, -1) + 1
                            counter[start_description] = gest_num
                            verified_annotations.append((annotation[0], f'Start_{start_description}_{gest_num}'))
                            verified_annotations.append((annotations[i + 1][0], f'Release_{end_description}_{gest_num}'))
                else:
                    print(f'Error: annotation mismatch, no Release_ annotation for {start_description} in time: '
                          f'{annotation[0]}, in the experiment: {self.experiment}')
            else:
                continue

        # remove bad annotations - if we have the same gesture in two annotations remove the first one
        good_annotations = []
        annotations_description = [annotation[1] for annotation in verified_annotations]
        for i, annotation in enumerate(verified_annotations):
            if annotation[1] not in annotations_description[i+1:]:
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

        if self. pipeline.segmentation_type == "discrete":
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
            fig.axes[i*max(num_repetitions_per_gesture.values())].set_ylabel('h', rotation=0, fontsize=18)
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
        segment_length_acc = floor(self.pipeline.segment_length_sec * self.pipeline.acc_sample_rate)
        num_emg_channels = self.raw_edf_emg.info['nchan']
        num_acc_channels = self.raw_edf_acc.info['nchan']

        segments_emg = np.empty(shape=(0, num_emg_channels, segment_length_emg), dtype=np.float16)
        segments_acc = np.empty(shape=(0, num_acc_channels, segment_length_acc), dtype=np.float16)
        labels = np.empty(shape=(0,), dtype=str)
        step_size = floor(self.pipeline.segment_step_sec * self.pipeline.emg_sample_rate)
        for emg_data, acc_data, label in self.annotations_data:
            # emg segmentation and labels creation
            for i in range(0, emg_data.shape[1] - segment_length_emg, step_size):
                curr_emg_segment = emg_data[:, i:i + segment_length_emg][np.newaxis, :, :]
                segments_emg = np.vstack((segments_emg, curr_emg_segment))
                labels = np.append(labels, label)
            # acc segmentation
            for i in range(0, acc_data.shape[1] - segment_length_acc, segment_length_acc):
                curr_acc_segment = acc_data[:, i:i + segment_length_acc][np.newaxis, :, :]
                segments_acc = np.vstack((segments_acc, curr_acc_segment))

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

    @staticmethod
    def norm_me(data: np.array, norm_type: str) -> np.array:
        if norm_type == 'zscore':
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
            data = (data - mean) / std
        elif norm_type == '01':
            min_ = np.min(data, axis=(1, 2), keepdims=True)
            max_ = np.max(data, axis=(1, 2), keepdims=True)
            data = (data - min_) / (max_ - min_)
        elif norm_type == '-11':
            min_ = np.min(data, axis=(1, 2), keepdims=True)
            max_ = np.max(data, axis=(1, 2), keepdims=True)
            data = 2 * (data - min_) / (max_ - min_) - 1
        elif 'quantile' in norm_type:
            quantiles = norm_type.split('_')[1].split('-')
            quantiles = [float(q) for q in quantiles]
            low_quantile = np.quantile(data, quantiles[0], axis=(1, 2), keepdims=True)
            high_quantile = np.quantile(data, quantiles[1], axis=(1, 2), keepdims=True)
            data = (data - low_quantile) / (high_quantile - low_quantile)
        elif norm_type == 'none':
            pass
        else:
            raise ValueError('Invalid normalization method for EMG data')
        return data

    def extract_features(self) -> None:
        feature_extractor = build_feature_extractor(self.pipeline.features_extraction_method)
        features = feature_extractor.extract_features(self.segments, **self.pipeline.features_extraction_params)
        self.features = features

    def normalize_features(self) -> np.array:
        self.features = self.norm_me(self.features, self.pipeline.features_norm)

    def get_dataset(self, include_synthetics = False) -> (np.array, np.array):
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
