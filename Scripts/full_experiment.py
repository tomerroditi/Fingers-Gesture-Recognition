import sklearn
import matplotlib.pyplot as plt
import numpy as np

from Source.fgr.data_manager import Recording_Emg_Live
from Source.streamer.data import Data
from Source.fgr.data_collection import Experiment
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.models import Net
from Source.fgr.utils import train_test_split_by_gesture
from Source.utils import save_model_and_pipeline


subject_num = 2
position_num = 1
trial_num = 0
session_num = 1
n_rep = 10

host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
data_collector = Data(host_name, port, timeout_secs=30, verbose=False)
exp = Experiment(subject_num=subject_num, position_num=position_num, session_num=session_num, trial_num=trial_num)
exp.run(data_collector=data_collector, n_repetitions=n_rep, img_sec=5, instruction_secs=2, relax_sec=0.5)

# create a dataset for model training
pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
rec = Recording_Emg_Live(data_collector.exg_data.T, data_collector.annotations, pipe)
dataset = rec.get_dataset()
data_train, data_val, labels_train, labels_val = train_test_split_by_gesture(dataset[0], labels=dataset[1], test_size=0.2)

# train a model
model = Net(num_classes=10, dropout_rate=0.1)
model.fit_model(data_train, labels_train, data_val, labels_val, num_epochs=50, batch_size=64, lr=0.001, l2_weight=0.0001)
model.evaluate_model(model.train_data, model.train_labels, cm_title='model train results')
model.evaluate_model(model.val_data, model.val_labels, cm_title='model validation results')

# save the model and the pipeline
save_model_and_pipeline(model, pipe, subject_num)

# run a live evaluation of the model
n_rep = 5
exp = Experiment(subject_num=subject_num, position_num=position_num, session_num=session_num, trial_num=trial_num + 1)
exp.run(data_collector=data_collector, pipeline=pipe, model=model, n_repetitions=n_rep, img_sec=5, instruction_secs=2, relax_sec=0.5 )

# confusion matrix - summary of the model performance in the live evaluation
true = []
pred = []
for key, val in exp.predictions.items():
    true.extend([key] * len(val))
    pred.extend(val)
sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(true, pred),
                                       display_labels=np.sort(np.unique(true))).plot(cmap='Blues')
plt.show()
print(f'accuracy: {sklearn.metrics.accuracy_score(true, pred)}')
