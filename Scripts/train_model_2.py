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

data_train, labels_train = rec_train.get_dataset()
data_test, labels_test = rec_test.get_dataset()
data_train, data_val, labels_train, labels_val = train_test_split_by_gesture(data_train, labels=labels_train,
                                                                             test_size=0.2)
# data_test_0, data_test, labels_test_0, labels_test = train_test_split_by_gesture(data_test, labels=labels_test,
#                                                                                         test_size=0.6)
#
# data_train = np.concatenate([data_train, data_test_0], axis=0)
# labels_train = np.concatenate([labels_train, labels_test_0], axis=0)

# rec_train.heatmap_visualization(10)
# rec_test.heatmap_visualization(5)

model = Net(num_classes=10 - len(discard), dropout_rate=0.4)
model.fit_model(data_train, labels_train, data_val, labels_val,  num_epochs=200, batch_size=64, lr=0.001,
                l2_weight=0.0001)
model.evaluate_model(model.train_data, model.train_labels, cm_title='model train results')
model.evaluate_model(model.val_data, model.val_labels, cm_title='model validation results')
model.evaluate_model(data_test, labels_test, cm_title='model test results')

# save the model and the pipeline, the saved pipeline is used to determine the preprocessing in the live experiment
save_model_and_pipeline(model, pipe, subject_num)

