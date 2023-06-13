import torch
import time

from Source.fgr.models import Real_Time_Predictor
from Source.fgr.pipelines import Data_Pipeline
from Source.streamer.data import Data
from Source.streamer.viz import Viz

# load a model
model_fp = r'C:\Users\YH006_new\Desktop\Fingers-Gesture-Recognition\data\071\model.pth'
model = torch.load(model_fp)

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
    verbose = True
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)
    data_collector.start()
    plotter = Viz(data_collector, plot_imu=False, plot_ica=False, ylim_exg=(-350, 350), update_interval_ms=50,
                  max_timeout=20)
    plotter.start()
    exit(0)

# configure the data pipeline - in future we will make sure it is saved a model attribute to make sure the data is
# processed the same way it was during training
pipeline = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)  # configure the data pipeline you would like to use (check pipelines module for more info)

# create a real time predictor
predictor = Real_Time_Predictor(model, data_streamer, pipeline, vote_over=70, max_timeout=20)

j = 0
time.sleep(5)  # let some data to aggregate in the data streamer
while data_streamer.is_connected:
    j += 1
    time.sleep(0.07)
    prediction, confidence = predictor.majority_vote_predict()
    if j == 20:
        print(f'{prediction}, confidence: {confidence}')
        j = 0
# data_streamer.save_data()




