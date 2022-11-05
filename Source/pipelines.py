from dataclasses import dataclass

data_files_folder = r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files'


@dataclass  # useful wrapper to make a class with only attributes and no methods (like a struct)
class Data_Pipeline:
    """this class is a dataclass that holds the parameters for data preprocessing"""
    segmentation_type: str = 'discrete'  # discrete or continuous
    sample_rate: int = 4000  # sample rate in HZ
    subsegment_duration: float = 1
    band_pass_freq: tuple = (40, 450)
    normalization_quantiles: tuple = (0.01, 0.99)


