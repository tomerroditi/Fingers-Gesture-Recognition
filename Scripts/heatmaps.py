# import numpy as np
# import torch
# import Source.fgr.models as models
#
# from Source.fgr.pipelines import Data_Pipeline
# from Source.fgr.data_manager import Data_Manager
# from warnings import simplefilter
# from pathlib import Path
# from importlib import reload
#
# # pipeline definition and data manager creation
# data_path = Path(r'I:/My Drive/finger gesture recognition/')
# pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
# subject = 1
# dm = Data_Manager([subject], pipeline)
# print(dm.data_info())
#
# dm.get_dataset(experiments=[f'{subject:03d}_*_*'])
# for subj in dm.subjects:
#     for rec in subj.recordings:
#         if rec.emg_data is not None:
#             rec.heatmap_visualization(10)


"""This script is used to train a model out of saved data files"""
import pickle
import numpy as np

from pathlib import Path
from Source.fgr.data_manager import Recording_Emg_Live
from Source.fgr.utils import train_test_split_by_gesture
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.models import Net
from Source.utils import save_model_and_pipeline

subject_num = 2
position_num = 1
session_num = 1
train_trial_num = 0
test_trial_num = 1


# load the data
path = Path().cwd().parent / 'data'
train_path = path / f'{subject_num:03d}/subject-{subject_num:03d}_position-{position_num:02d}_session' \
                    f'-{session_num:02d}_trial-{train_trial_num:02d}.pkl'
test_path = path / f'{subject_num:03d}/subject-{subject_num:03d}_position-{position_num:02d}_session' \
                  f'-{session_num:02d}_trial-{test_trial_num:02d}.pkl'
with open(train_path, 'rb') as f:
    train_data = pickle.load(f)
annotations_train = train_data['annotations']
data_train = train_data['emg'].T

with open(test_path, 'rb') as f:
    test_data = pickle.load(f)
annotations_test = test_data['annotations']
data_test = test_data['emg'].T

# discard unwanted gestures
discard = []
annotations_train = [a for a in annotations_train if all([d not in a[2] for d in discard])]
annotations_val = [a for a in annotations_test if all([d not in a[2] for d in discard])]

pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124, features_norm='none')
rec_train = Recording_Emg_Live(data_train, annotations_train, pipe)
rec_test = Recording_Emg_Live(data_test, annotations_val, pipe)

rec_train.heatmap_visualization(10)
rec_test.heatmap_visualization(5)