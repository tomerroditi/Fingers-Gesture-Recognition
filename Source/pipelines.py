from dataclasses import dataclass, field
from pathlib import Path


@dataclass  # useful wrapper to make a class with only attributes and no methods (like a struct)
class Data_Pipeline:
    """this class is a dataclass that holds the parameters for data preprocessing"""
    base_data_files_path: Path = Path(r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files')
    segmentation_type: str = 'discrete'  # discrete or continuous
    emg_sample_rate: int = 4000  # sample rate in HZ
    acc_sample_rate: int = 4000  # sample rate in HZ
    segment_length_sec: float = 0.2
    emg_high_freq: float = 400
    emg_low_freq: float = 20
    emg_notch_freq: float = 50
    normalization_quantiles_segments: tuple = None  # (lower, upper) quantiles for normalization of segments
    normalization_quantiles_features: tuple = (0.01, 0.99)  # (lower, upper) quantiles for normalization of features
    features_extraction_method: str = 'RMS'  # name of the method to extract features from segments
    features_extraction_params: dict = field(default_factory=dict)  # parameters for the features extraction method
    num_repetition_hmm: int = 6  # number of repetitions for the HMM model


@dataclass
class Model_Pipeline:
    """this class is a dataclass that holds the parameters for model training"""
    def __init__(self, model_type: str, model_params: dict, train_params: dict):
        self.model_type = model_type
