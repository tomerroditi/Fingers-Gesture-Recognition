import os.path
import Source.fgr.data_manager as dm
import torch
import pickle

from pathlib import Path
from Source.streamer.data import Data
from Source.fgr.data_collection import Experiment
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.models import Net, pre_training_utils

subject_num = 10
position_num = 1
trial_num = 0
session_num = 1
n_rep = 10

host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
data_collector = Data(host_name, port, timeout_secs=30, verbose=False)
exp = Experiment(subject_num=subject_num, position_num=position_num, session_num=session_num, trial_num=trial_num)
exp.run(data_collector=data_collector)

pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
rec = dm.Recording_Emg_Live(data_collector.exg_data.T, data_collector.annotations, pipe)

dataset = rec.get_dataset()
data_train, data_val, labels_train, labels_val = pre_training_utils.train_test_split_by_gesture(dataset[0], labels=dataset[1])
data_train = data_train.reshape(data_train.shape[0], 1, 4, 4)  # reshape to fit the CNN input
data_val = data_val.reshape(data_val.shape[0], 1, 4, 4)  # reshape to fit the CNN input

model = Net(num_classes=10, dropout_rate=0.1)
model.fit_model(data_train, labels_train, data_val, labels_val,  num_epochs=50, batch_size=64, lr=0.001, l2_weight=0.0001)
model.evaluate_model(model.train_data, model.train_labels, cm_title='model results')

# %% set and train a model (cv or not)
model_name = 'model_0'
path = Path('../data/models')
while os.path.exists(path / f'{subject_num}/{model_name}.pth'):
    model_name = f'{model_name.split("_")[0]}_{int(model_name.split("_")[1])+1}'
path = path / f'{subject_num}/{model_name}.pth'
torch.save(model, path)
print(f'model saved to: {path}')
# save the pipeline
with open(Path('../data/models') / f'{subject_num}/{model_name}_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)
print(f'pipeline saved to: {Path("../data/models") / f"{subject_num}/{model_name}_pipeline.pkl"}')
