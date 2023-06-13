import os.path
import pickle
import numpy as np
import time

from datetime import datetime

from pathlib import Path
from Source.streamer.data import Data
from Source.streamer.viz import Viz
from Source.fgr.data_collection import Experiment
from Source.fgr import models
from Source.fgr.data_manager import Recording_Emg
from Source.fgr.pipelines import Data_Pipeline


""" RUN PARAMETERS """

host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
n_repetitions = 2  # Number of repetitions of each gesture during data collection
signal_check = False  # View real-time signals as signal quality check? Note: running this precludes PsychoPy (idk why) so run it once with True then run again with False

"""SIGNAL CHECK """

if signal_check:
    timeout = None
    verbose = True
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)
    data_collector.start()
    plotter = Viz(data_collector, plot_imu=False, plot_ica=False, ylim_exg=(-350, 350), update_interval_ms=50,
                  max_timeout=20)
    plotter.start()
    exit(0)

""" DATA COLLECTION """

start_time = datetime.now()

timeout = 30  # if streaming is interrupted for this many seconds or longer, terminate program
verbose = False  # if to print to console BT packet summary (as sanity check) upon each received packet
data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)

# PsychoPy experiment
data_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent / 'data')
exp = Experiment()
file_path = exp.run(data_collector, data_dir, n_repetitions=n_repetitions, fullscreen=False, img_secs=3, screen_num=0)

calibration_time = datetime.now()
print('Data collection complete. Beginning model creation...')

""" MODEL CREATION """
time.sleep(1)
# Get data and labels
pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
rec_obj = Recording_Emg([Path(file_path)], pipe)
X, labels = rec_obj.get_dataset()

# Train models
n_models = min(5, n_repetitions)
n_classes = 10
model = models.Net(num_classes=n_classes, dropout_rate=0.1)
models, accu_vals = model.cv_fit_model(X, labels, num_epochs=150, batch_size=64, lr=0.001, l2_weight=0.0001,
                                       num_folds=n_models, plot_cm=True)

# Evaluate models
print(f'model average accuracy: {np.mean(accu_vals)}')
print(f'best model test accuracy: {np.max(accu_vals) * 100:0.2f}%')

end_time = datetime.now()
calibration_duration = calibration_time - start_time
train_duration = end_time - calibration_time
process_duration = end_time - start_time
print('Training process complete.')
print(f'Calibration took {calibration_duration}')
print(f'Training took {train_duration}')
print(f'Whole process took {process_duration}')