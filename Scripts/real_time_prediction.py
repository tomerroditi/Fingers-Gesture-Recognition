"""
This script demonstrates how to use the real time predictor.
"""

import torch
import time
import matplotlib.pyplot as plt

from Source.fgr.models import Real_Time_Predictor
from Source.fgr.pipelines import Data_Pipeline
from Source.streamer.data import Data
from Source.streamer.viz import Viz

# load a model
subject_num = 1
model = torch.load(r'C:\Users\YH006_new\Desktop\Fingers-Gesture-Recognition\data\models\model_31.pth')

# conect to a streamer
signal_check = False  # set to True to check the signal before running the real time predictor
host_name = "127.0.0.1"
port = 20001
timeout = 20
verbose = False
data_streamer = Data(host_name, port, verbose=verbose, timeout_secs=timeout, save_as='test.edf')
data_streamer.start()

# todo: make the signal check a util function
if signal_check:
    timeout = None
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=False)
    data_collector.start()
    plotter = Viz(data_collector, plot_imu=False, plot_ica=False, ylim_exg=(-350, 350), update_interval_ms=50,
                  max_timeout=20)
    plotter.start()
    exit(0)

# configure the data pipeline - in future we will make sure it is saved a model attribute to make sure the data is
# processed the same way it was during training
pipeline = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
# configure the data pipeline you would like to use (check pipelines module for more info)

# create a real time predictor
predictor = Real_Time_Predictor(model, data_streamer, pipeline, vote_over=70, max_timeout=20)

j = 0
time.sleep(5)  # let some data to aggregate in the data streamer
last_pred = None
sleep_time = 0.05
while data_streamer.is_connected:
    j += 1
    time.sleep(sleep_time)
    prediction, confidence = predictor.majority_vote_predict()
    # plot the prediction
    if confidence > 0.5:
        if prediction == 'loading predictions...' or last_pred == prediction:
            sleep_time = 0.05
            pass
        else:
            sleep_time = 0  # plotting takes ~ 0.05 - 0.06 seconds
            last_pred = prediction
            # plot a text box with the prediction name
            plt.text(0.5, 0.5, prediction, horizontalalignment='center', verticalalignment='center', fontsize=50)
            plt.show()
    if j == 30:
        print(f'{prediction}, confidence: {confidence}')
        j = 0
# data_streamer.save_data()




