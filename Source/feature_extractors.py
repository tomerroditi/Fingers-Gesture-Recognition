import numpy as np
from abc import ABC, abstractmethod


class Feature_Extractor(ABC):
    @abstractmethod
    def extract_features(self, segments: (np.array, np.array), **kwargs) -> np.array:
        """
        This function is responsible for extracting features from the segments.
        inputs:
            segments: tuple of (emg_segments, acc_segments), emg_segments is a 3d array of shape (num_segments,
                      num_channels, segment_length) where num_channels is constant 16, acc_segments is a 3d array of
                      shape (num_segments, num_channels, segment_length) where num_channels is constant 3 (x, y, z - in
                      that order)
            kwargs: not used
        """
        pass


class RMS_Feature_Extractor(Feature_Extractor):
    def extract_features(self, segments: (np.array, np.array), **kwargs) -> np.array:
        """
        extract the RMS of each emg channel and reshape into a 4x4 array.
        input:
            segments: tuple of (emg_segments, acc_segments), emg_segments is a 3d array of shape (num_segments,
            num_channels, segment_length) where num_channels is constant 16, acc_segments is a 3d array of shape
            (num_segments, num_channels, segment_length) where num_channels is constant 3 (x, y, z - not in that order)
            kwargs: not used
        output:
            features: array of shape (num_segments, 4, 4)
        """
        emg_data, acc_data = segments
        features = np.sqrt(np.mean(np.square(emg_data), axis=2))
        return features


"""Builder"""
extractors = {"RMS": RMS_Feature_Extractor()}


def build_feature_extractor(method: str) -> Feature_Extractor:
    try:
        extractor = extractors[method]
        return extractor
    except KeyError:
        raise ValueError(f"Invalid method name: {method}")
