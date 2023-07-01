import os.path
import torch
import sklearn
import numpy as np
import pickle

from pathlib import Path
from Source.streamer.data import Data
from Source.fgr.data_collection import Experiment

subject_num = 10
position_num = 1
trial_num = 0
session_num = 1
n_rep = 10
model_num = 0

# load the model
path_model = Path(f'../data/models/{subject_num}/model_{model_num}.pth')
path_pipeline = Path(f'../data/models/{subject_num}/model_{model_num}_pipeline.pkl')
model = torch.load(path_model)
with open(path_pipeline, 'rb') as f:
    pipeline = pickle.load(f)

# run the experiment while predicting the labels
host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
data_collector = Data(host_name, port, timeout_secs=30, verbose=False)
exp = Experiment(subject_num=subject_num, position_num=position_num, session_num=session_num, trial_num=trial_num)
exp.run(data_collector=data_collector, pipeline=pipeline, model=model)

# confusion matrix
true = []
pred = []
for key, val in exp.predictions.items():
    true.extend([key] * len(val))
    pred.extend(val)
cm = sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(true, pred),
                                            display_labels=np.sort(np.unique(true))).plot()
print(f'accuracy: {sklearn.metrics.accuracy_score(true, pred)}')
