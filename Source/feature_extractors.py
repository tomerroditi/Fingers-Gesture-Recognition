import numpy as np
from abc import ABC, abstractmethod


class Feature_Extractor(ABC):
    @abstractmethod
    def extract_features(self, segments: (np.array, np.array), **kwargs) -> np.array:
        pass


class RMS_Feature_Extractor(Feature_Extractor):
    def extract_features(self, segments: (np.array, np.array), **kwargs) -> np.array:
        """extract the RMS of each emg channel and reshape into a 4x4 array.
        this is last year's method"""
        emg_data, acc_data = segments
        features = np.sqrt(np.mean(np.square(emg_data), axis=2))
        features = np.reshape(features, (features.shape[0], 4, 4))
        return features


"""Builder"""
extractors = {"RMS": RMS_Feature_Extractor()}


def build_feature_extractor(method: str) -> Feature_Extractor:
    try:
        extractor = extractors[method]
        return extractor
    except KeyError:
        raise ValueError(f"Invalid method name: {method}")
