import sys
sys.path.append('../')

from Source.fgr.data_manager import Recording_Emg
from Source.fgr.pipelines import Data_Pipeline

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


data_dir = '../../data/test'
file_paths = [Path(os.path.abspath(os.path.join(data_dir, data))) for data in os.listdir(data_dir)]
print(file_paths)
#pipeline object
pipeline = Data_Pipeline()

recording = Recording_Emg(files_path=file_paths, pipeline=pipeline)
X, labels = recording.get_dataset()
print(X.shape)