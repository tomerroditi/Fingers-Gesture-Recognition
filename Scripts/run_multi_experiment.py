import os.path
import numpy as np

from datetime import datetime
from pathlib import Path
from Source.streamer.data import Data
from Source.streamer.viz import Viz
from Source.fgr.data_collection import Experiment
from Source.fgr import models
from Source.fgr.data_manager import Recording_Emg
from Source.fgr.pipelines import Data_Pipeline


# data collector params
host_name = "127.0.0.1"  # IP address from which to receive data
port = 20001  # Local port through which to access host
data_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent / 'data')

# model initialization
n_classes = len(os.listdir('images'))
model = models.Net(num_classes=n_classes, dropout_rate=0.1)

# experiment params
exp_n_repetitions = [2, 2, 3, 2]
X = None
labels = None


for j in range(len(exp_n_repetitions)):
    """ DATA COLLECTION """
    start_time = datetime.now()

    timeout = 30  # if streaming is interrupted for this many seconds or longer, terminate program
    verbose = False  # if to print to console BT packet summary (as sanity check) upon each received packet
    data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)

    # PsychoPy experiment
    exp = Experiment()
    file_path = exp.run(data_collector, data_dir, n_repetitions=exp_n_repetitions[j], fullscreen=False, img_secs=3,
                        screen_num=0, exp_num=j)

    calibration_time = datetime.now()
    print('Data collection complete. Beginning model creation...')

    """ MODEL CREATION """

    # Get data and labels and extend the dataset
    pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
    rec_obj = Recording_Emg([Path(file_path)], pipe)
    curr_X, curr_labels = rec_obj.get_dataset()
    if X is None:
        X = curr_X
        labels = curr_labels
    else:
        X = np.concatenate((X, curr_X))
        labels = np.concatenate((labels, curr_labels))

    # Train models with the full dataset
    models, accuracies = model.cv_fit_model(X, labels,
                                            num_epochs=200,
                                            batch_size=64,
                                            lr=0.001,
                                            l2_weight=0.0001,
                                            num_folds=min(5, exp_n_repetitions[j]))

    # Evaluate models
    mean_accuracy = np.mean(accuracies)
    print(f'Round {j + 1} results:')
    print(f'models average accuracy: {mean_accuracy * 100:0.2f}%')
    print(f'best model val accuracy: {np.max(accuracies) * 100:0.2f}%')
    if mean_accuracy >= 0.9:
        print('Subject model has reached 90% accuracy or above. \nTerminating program...')
        break
