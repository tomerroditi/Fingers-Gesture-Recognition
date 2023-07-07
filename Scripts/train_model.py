"""This script is used to train a model out of saved data files"""
import pickle

from pathlib import Path
from Source.fgr.data_manager import Recording_Emg_Live
from Source.fgr.utils import train_test_split_by_gesture
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.models import Net
from Source.utils import save_model_and_pipeline

subject_num = 1
position_num = 1
session_num = 1
trial_num = 0


# load the data
path = Path().cwd().parent / 'data'
path = path / f'{subject_num:03d}/subject-{subject_num:03d}_position-{position_num:02d}_session-{session_num:02d}_trial-{trial_num:02d}.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
annotations = data['annotations']
data = data['emg'].T

# discard unwanted gestures
discard = []
annotations = [a for a in annotations if all([d not in a[2] for d in discard])]

pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
rec = Recording_Emg_Live(data, annotations, pipe)

dataset = rec.get_dataset()
data_train, data_val, labels_train, labels_val = train_test_split_by_gesture(dataset[0], labels=dataset[1], test_size=0.2)

model = Net(num_classes=10 - len(discard), dropout_rate=0.1)
model.fit_model(data_train, labels_train, data_val, labels_val,  num_epochs=50, batch_size=64, lr=0.001, l2_weight=0.0001)
model.evaluate_model(model.train_data, model.train_labels, cm_title='model train results')
model.evaluate_model(model.val_data, model.val_labels, cm_title='model validation results')

# save the model and the pipeline, the saved pipeline is used to determine the preprocessing in the live experiment
save_model_and_pipeline(model, pipe, subject_num)

