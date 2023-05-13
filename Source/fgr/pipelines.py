import torch.nn

from dataclasses import dataclass, field
from pathlib import Path


@dataclass  # useful wrapper to make a class with only attributes and no methods (like a struct)
class Data_Pipeline:
    """this class is a dataclass that holds the parameters for data preprocessing"""
    base_data_files_path: Path = Path(r'I:\My Drive\finger gesture recognition')
    max_gesture_duration: float = 7  # the supposed maximum duration of a gesture in seconds (depends on the
    # protocol), it is used to detect corrupted annotations.
    annotation_delay_start: float = 0.1  # the delay between the annotation and the start of the gesture in seconds,
    # when we segment the data (in discrete mode) we will start the segment from the annotation time + this delay
    annotation_delay_end: float = 0.1  # the delay between the annotation and the end of the gesture in seconds,
    # when we segment the data (in discrete mode) we will end the segment at the annotation time - this delay
    segmentation_type: str = 'discrete'  # discrete or continuous segmentation protocol
    # todo: check if the fs can be retrieved from the edf file
    emg_sample_rate: int = 4000  # sample rate in HZ
    acc_sample_rate: int = 4000  # sample rate in HZ
    segment_length_sec: float = 0.6  # the length of the segments in seconds
    segment_step_sec: float = 0.075  # step between segments in seconds
    emg_high_freq: float = 400  # high cut-off frequency for the emg bandpass filter
    emg_low_freq: float = 20  # low cut-off frequency for the emg bandpass filter
    emg_notch_freq: float = 50  # notch frequency for the emg notch filter
    emg_norm: str = 'none'  # none, zscore, 01, -11, quantile_#-# --> normalization method for the emg data
    acc_norm: str = 'none'  # none, zscore, 01, -11, quantile_#-# --> normalization method for the acc data
    features_norm: str = 'max'  # none, zscore, 01, -11, quantile_#-# --> normalization method for the features
    features_extraction_method: str = 'RMS'  # name of the method to extract features from segments, to see available
    # methods see the "extractors" dictionary in the data_manager.py file.
    features_extraction_params: dict = field(default_factory=dict)  # parameters for the features extraction method


@dataclass
class Model_Pipeline:
    """this class is a dataclass that holds the parameters for model training"""
    # TODO: refactor this class for a more convenient way to use it
    def __init__(self, model: str, model_params: dict, train_params: dict):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.num_epochs: int = 200
        self.batch_size: int = 64


