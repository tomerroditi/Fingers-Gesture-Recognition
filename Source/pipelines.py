from dataclasses import dataclass, field
from pathlib import Path


@dataclass  # useful wrapper to make a class with only attributes and no methods (like a struct)
class Data_Pipeline:
    """this class is a dataclass that holds the parameters for data preprocessing"""
    base_data_files_path: Path = Path(r'C:\Users\tomer\Documents\data sets\finger gesture recognition')
    max_gesture_duration: float = 8  # the supposed maximum duration of a gesture in seconds (depends on the protocol)
    annotation_delay_start: float = 1
    annotation_delay_end: float = 1
    segmentation_type: str = 'discrete'  # discrete or continuous
    emg_sample_rate: int = 4000  # sample rate in HZ
    acc_sample_rate: int = 4000  # sample rate in HZ
    segment_length_sec: float = 0.4
    segment_step_sec: float = 0.1  # step between segments in seconds
    emg_high_freq: float = 400
    emg_low_freq: float = 20
    emg_notch_freq: float = 50
    emg_norm: str = 'none'  # none, zscore, 01, -11, quantile_#-#
    acc_norm: str = 'none'  # none, zscore, 01, -11, quantile_#-#
    features_norm: str = 'none'  # none, zscore, 01, -11, quantile_#-#
    features_extraction_method: str = 'RMS'  # name of the method to extract features from segments
    features_extraction_params: dict = field(default_factory=dict)  # parameters for the features extraction method


@dataclass
class Model_Pipeline:
    """this class is a dataclass that holds the parameters for model training"""
    def __init__(self, model_type: str, model_params: dict, train_params: dict):
        self.model_type = model_type
