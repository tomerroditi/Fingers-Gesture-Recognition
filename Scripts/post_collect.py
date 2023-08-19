"""
This script is used to live test a model performance on a subject.
It is using a trained model to predict the labels of the gestures that are presented to the subject.
it saves the recorded data as pickle files (new online recordings) and as edf files.

you may modify the parameters at the beginning of the script to your needs.
subject_num, position_num, session_num, trial_num - determines the saved file name and location.
model_num - each subject has several models, this parameter determines which model to use.
"""

import torch
import sklearn
import numpy as np
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from Source.streamer.data import Data
from Source.fgr.data_collection import Experiment

subject_num = 1
position_num = 1
session_num = 1
trial_num = 0
n_rep = 1
model_num = 3

# load the model
path_model = Path(f'../models/{subject_num:03d}/model_{model_num}.pth')
path_pipeline = Path(f'../models/{subject_num:03d}/model_{model_num}_pipeline.pkl')
model = torch.load(path_model)
with open(path_pipeline, 'rb') as f:
    pipeline = pickle.load(f)

# run the experiment while predicting the labels
host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
data_collector = Data(host_name, port, timeout_secs=30, verbose=False)
exp = Experiment(subject_num=subject_num, position_num=position_num, session_num=session_num, trial_num=trial_num)
exp.run(data_collector=data_collector, pipeline=pipeline, model=model, n_repetitions=n_rep, img_sec=5, instruction_secs=2, relax_sec=0.5 )

# confusion matrix
true = []
pred = []
for key, val in exp.predictions.items():
    true.extend([key] * len(val))
    pred.extend(val)
sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(true, pred), display_labels=np.sort(np.unique(true))).plot(cmap='Blues')
plt.show()
print(f'accuracy: {sklearn.metrics.accuracy_score(true, pred)}')

