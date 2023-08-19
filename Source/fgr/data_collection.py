import collections
import os
import pickle
import numpy as np
import sklearn
import matplotlib.pyplot as plt

from os.path import dirname, join, abspath
from psychopy import visual, core, event
from psychopy.hardware import keyboard
from pathlib import Path
from .data_manager import Recording_Emg_Live
from ..streamer.data import Data


class Experiment:
    """This class is used to run a PsychoPy experiment for data collection."""

    def __init__(self, subject_num: int = 0, session_num: int = 0, position_num: int = 0, trial_num: int = 0):
        """Initialize the experiment.

        Parameters
        ----------
        subject_num : int, optional
            Subject number. The default is 0.
        session_num : int, optional
            Session number. The default is 0.
        position_num : int, optional
            Position number. The default is 0.
        trial_num : int, optional
            Trial number. The default is 0.
        """
        self.subject_num = subject_num
        self.session_num = session_num
        self.position_num = position_num
        self.trial_num = trial_num
        self.predictions = collections.defaultdict(list)

        self.data = None
        self.exit = False  # Flag to exit the experiment
        self.img_dir = join(dirname(dirname(dirname(abspath(__file__)))), 'images')
        self.file_name = f"subject-{subject_num:03d}_position-{position_num:02d}_" \
                         f"session-{session_num:02d}_trial-{trial_num:02d}"
        self.keyboard = keyboard.Keyboard(backend='iohub')
        self.win = visual.Window(fullscr=True, screen=0, color=[-1, -1, -1], units='height', monitor='default')

    def run(self, data_collector: Data = None, n_repetitions: int = 10, img_sec: float = 5,
            instruction_secs: float = 4, relax_sec: float = 0.5, pipeline=None, model=None):
        """
        Run the experiment. in case rec_obj and model are provided each performed gesture will be classified by the
        model and will give us an instant feedback.

        Parameters
        ----------
        data_collector : Data
            object to send annotations to.
        n_repetitions : int, optional
            Number of repetitions of each gesture during data collection. The default is 10.
        img_sec : float, optional
            Number of seconds to display each image. The default is 5.
        instruction_secs : float, optional
            Number of seconds to display the instruction text. The default is 4.
        relax_sec : float, optional
            Number of seconds to relax between gestures. The default is 0.5.
        pipeline : Data_Pipeline, optional
            a data preprocessing pipeline. The default is None. used for testing a model.
        model : torch.nn.Module, optional
            a trained model. The default is None. used for testing a model.
        """
        self.data = data_collector
        if self.data is not None:  # set a path to the save data in case we are recording
            path = Path(__file__).parent.parent.parent / f'data/{self.subject_num:03d}'
            if not path.exists():
                path.mkdir(parents=True)
            self.data.save_as = str(path / f'{self.file_name}.edf')
            # check if file name already exists, if so add to the trial number until it doesn't
            while os.path.exists(self.data.save_as) or os.path.exists(self.data.save_as[:-4] + '.pkl'):
                trial_num = int(self.data.save_as.split('_')[-1].split('-')[-1].split('.')[0]) + 1
                self.file_name = '_'.join(self.file_name.split('_')[:-1]) + f'_trial-{trial_num:02d}'
                self.data.save_as = str(path / f'{self.file_name}.edf')
            self.data.start()  # start recording data

        # welcome screen
        self.welcome_screen()

        # collect data
        image_files = [f for f in os.listdir(self.img_dir) if f.endswith('.JPG') or f.endswith('.jpg')]
        for image_file in image_files:
            self.gesture_screen(image_file, n_repetitions, img_sec, instruction_secs, relax_sec, pipeline, model)
            if self.exit:
                break

        self.win.close()  # close the window of the experiment
        self.save_data()  # save the collected data to a pickle file
        self.data.stop()  # stop recording data, and save an edf file

        if self.predictions:
            self.plot_predictions_cm()

    def trigger(self, msg, verbose: bool = True):

        if self.data is not None:
            self.data.add_annotation(msg)
            if verbose:
                print(f'TRIGGER: {self.data.annotations[-1]}')
        elif verbose:
            print(f'TRIGGER: {msg}')

    def welcome_screen(self):
        welcome_text = visual.TextStim(
                win=self.win, name='welcome_text',
                # edd to the text below which position we are in
                text='A series of images will be shown on screen.\n\n\n'
                     f'You are in position {self.position_num}.\n\n\n'
                     'Perform the gesture only when\n"Perform gesture"\nis written above the image.\n\n\n'
                     'Relax your arm between gestures.\n\n\n'
                     '(Press any key when ready.)', pos=(0, 0), height=0.04)

        welcome_text.draw()
        self.win.flip()
        event.waitKeys()  # Wait for a key press to start the experiment

    def gesture_screen(self, image_file, n_repetitions, img_sec, instruction_secs, relax_sec, pipeline, model):
        instruct_text = 'When "Perform gesture" appears, perform this gesture:'
        perform_text = 'Perform gesture'
        relax_text = 'Relax arm'
        text_kwargs = {'pos': (0, 0.3), 'height': 0.04}

        image_path = os.path.join(self.img_dir, image_file)
        image_name = image_file.split('.')[0]
        annotation = f"{image_name}_{self.subject_num:03d}_{self.session_num}_{self.position_num}_{self.trial_num}"

        image = visual.ImageStim(self.win, image=image_path, size=0.5)
        text_instruction = visual.TextStim(self.win, text=instruct_text, **text_kwargs)
        text_perform = visual.TextStim(self.win, text=perform_text, color=(0.2, 1, 0.2), **text_kwargs)
        text_relax = visual.TextStim(self.win, text=relax_text, **text_kwargs)

        # instruction block
        text_instruction.draw()
        image.draw()
        self.win.flip()
        core.wait(instruction_secs)

        for i in range(n_repetitions):
            # perform gesture block
            text_perform.draw()
            image.draw()
            self.win.flip()
            self.trigger(f"Start_{annotation}_{i:02d}")
            core.wait(img_sec)
            self.trigger(f"Release_{annotation}_{i:02d}")

            if pipeline is not None and model is not None:
                # relax arm block
                text_relax.draw()
                image.draw()
                self.win.flip()
                self.predict_block(pipeline, model, image_name, text_kwargs)

            # relax arm block
            text_relax.draw()
            image.draw()
            self.win.flip()
            core.wait(relax_sec)

            # check for escape key press and end experiment if pressed
            if self.keyboard.getKeys(keyList=["escape"]):
                self.save_data()
                self.win.close()
                self.exit = True
                return

    def predict_block(self, pipeline, model, image_name, text_kwargs):
        """Predict the performed gesture and give a feedback to the user."""
        # get the relevant data
        emg_data, annotations = self.get_last_gesture_data(pipeline.emg_sample_rate)
        # create a recording object
        rec = Recording_Emg_Live(emg_data.T, annotations, pipeline)
        dataset = rec.get_dataset()  # (data, labels)
        predictions = model.classify(dataset[0])
        # majority voting
        counter = collections.Counter(predictions)
        majority = counter.most_common(1)[0][0]
        confidence = counter[majority] / len(predictions)
        self.predictions[image_name].append(majority)
        # give a feedback to the user
        text = 'Correct' if majority == image_name else 'Wrong'
        text = f'{text} \n\n\npredicted - {majority} \n\n\nconfidence - {confidence}'
        text = visual.TextStim(self.win, text=text, **text_kwargs)
        text.draw()
        self.win.flip()
        core.wait(2)

    def get_last_gesture_data(self, sr) -> (np.ndarray, list[(float, float, str)]):
        """Chop the data to the relevant part."""
        annotations = self.data.annotations[-2:]
        start_time = annotations[0][0]
        duration = annotations[1][0] - start_time
        start_idx = int((start_time - 5) * sr)
        emg_data = self.data.exg_data[start_idx:, :]
        start_time = 5
        end_time = start_time + duration
        annotations = [(start_time, annotations[0][1], annotations[0][2]),
                       (end_time, annotations[1][1], annotations[1][2])]
        return emg_data, annotations

    def save_data(self):
        """Save the data to a pickle file and to an edf file."""
        if self.data is None:
            pass
        else:
            # save data to pickle file
            my_dict = {'emg': self.data.exg_data, 'annotations': self.data.annotations}
            my_path = Path(self.data.save_as)
            with open(my_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(my_dict, f)

    def plot_predictions_cm(self):
        """Plot the confusion matrix of the predictions and print the accuracy."""
        true = []
        pred = []
        for key, val in self.predictions.items():
            true.extend([key] * len(val))
            pred.extend(val)
        sklearn.metrics.ConfusionMatrixDisplay(sklearn.metrics.confusion_matrix(true, pred),
                                               display_labels=np.sort(np.unique(true))).plot(cmap='Blues')
        plt.show()
        print(f'accuracy: {sklearn.metrics.accuracy_score(true, pred)}')
