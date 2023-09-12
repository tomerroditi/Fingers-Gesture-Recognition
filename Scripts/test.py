import sys
sys.path.append('../')

import numpy as np
import torch
import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager
from warnings import simplefilter
from pathlib import Path
from importlib import reload

# ignore some warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

# pipeline definition and data manager creation
data_path = Path('../../data/doi_10')
pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
subject = 1
dm = Data_Manager([subject], pipeline)
print(dm.data_info())

# extract datasets from the data manager - labels format: "<subject>_<session>_<position>_<gesture>_<iteration_number>"
# dataset = dm.get_dataset(experiments=[f'{subject:03d}_*_*'])

# data = dataset[0]
# labels = dataset[1]

# # train test split
# data_train, data_test, labels_train, labels_test = models.pre_training_utils\
#     .train_test_split_by_gesture(data, labels=labels, test_size=0.2)

# # reshape the data to match the model architecture
# data_train = data_train.reshape(data_train.shape[0], 1, 4, 4)  # reshape to fit the CNN input

