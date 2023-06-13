import numpy as np
import multiprocessing
import time

from torch import nn
from pathlib import Path
from Source.streamer.data import Data
from Source.fgr.data_collection import Experiment
from Source.fgr.data_manager import Recording_Emg
from Source.fgr.pipelines import Data_Pipeline

def data_collection(host_name: str, port: int, data_dir: str, shared_dict: dict, lock: multiprocessing.Lock):
    """ DATA COLLECTION """
    timeout = 20  # if streaming is interrupted for this many seconds or longer, terminate program
    verbose = False  # if to print to console BT packet summary (as sanity check) upon each received packet
    j = 0
    while True:
        j += 1
        n = 8 if j == 1 else 1
        data_collector = Data(host_name, port, timeout_secs=timeout, verbose=verbose)

        # PsychoPy experiment
        exp = Experiment()
        file_path = exp.run(data_collector, data_dir, n_repetitions=n, fullscreen=False, img_secs=5,
                            screen_num=0, exp_num=j)

        time.sleep(5)
        # Get data and labels and extend the dataset
        pipe = Data_Pipeline(emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124)
        rec_obj = Recording_Emg([Path(file_path)], pipe)
        curr_X, curr_labels = rec_obj.get_dataset()

        lock.acquire()
        try:
            if shared_dict.get('accuracy', 0) < 0.9:
                if 'X' not in shared_dict or shared_dict['X'] is None:
                    shared_dict['X'] = curr_X
                    shared_dict['labels'] = curr_labels
                else:
                    shared_dict['X'] = np.concatenate((shared_dict['X'], curr_X))
                    shared_dict['labels'] = np.concatenate((shared_dict['labels'], curr_labels))
                shared_dict['num_runs'] = shared_dict.get('num_runs', 0) + 1
            else:
                shared_dict['last_X'] = curr_X
                shared_dict['last_labels'] = curr_labels
                shared_dict['terminate'] = True
                print('Subject model has reached 90% accuracy or above. Terminating data collection...')
                break

            # Update file paths in shared dict
            shared_dict['has_new_data'] = True
        finally:
            lock.release()


def train_model(model: nn.Module, shared_dict: dict, lock: multiprocessing.Lock):
    while True:
        lock.acquire()
        try:  # check shared dict values and extract data if new data is available or termination is requested
            has_new_data = shared_dict.get('has_new_data', False)
            terminate = shared_dict.get('terminate', False)
            if has_new_data or terminate:
                data = shared_dict['X']
                labels = shared_dict['labels']
                shared_dict['has_new_data'] = False  # reset flag
            if terminate:
                last_X = shared_dict['last_X']
                last_labels = shared_dict['last_labels']
        finally:
            lock.release()

        if has_new_data:
            print('Beginning model training...')
            # Train models with the current dataset
            models, accuracies = model.cv_fit_model(data, labels,
                                                    num_epochs=200,
                                                    batch_size=64,
                                                    lr=0.001,
                                                    l2_weight=0.0001,
                                                    num_folds=min(5, shared_dict['num_runs'] * 2),
                                                    plot_cm=True)

            # Evaluate models
            mean_accuracy = np.mean(accuracies)
            lock.acquire()
            try:
                shared_dict['accuracy'] = mean_accuracy
            finally:
                lock.release()
            print(f'Model average accuracy: {mean_accuracy * 100:0.2f}%')

        # Check termination flag
        elif terminate:
            model.fit_model(data, labels, val_data=last_X, val_labels=last_labels, num_epochs=200,
                            batch_size=64, lr=0.001, l2_weight=0.0001)
            model.evaluate_model(last_X, last_labels, plot_cm=True,
                                 cm_title='Subject Model Results (Test Data)')
            break
        else:
            time.sleep(10)  # check for new data every 10 seconds