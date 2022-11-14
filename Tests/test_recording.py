import pytest
import numpy as np
from Source.recording import Recording
from Source.pipelines import Data_Pipeline

file_paths = [
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos1_001_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_002\session_1\GR_pos2_002_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_003\session_1\GR_pos3_003_S1_Recording_00_SD_edited.edf'
    ]
exp_names = ['001_1_1', '002_1_2', '003_1_3']
general_exp_names = ['*_1_1', '002_*_*', '003_1_*']


@pytest.fixture(scope = 'module', params = zip(file_paths, exp_names, general_exp_names))
def my_file_path_and_exp_name(request):
    return request.param

# TODO: finish the fixture below to return a recording object
@pytest.fixture(scope = 'session')
def my_recording(monkeypatch):
    num_electrodes = 16
    # TODO: correct the annotations to be the same as in the files we work with
    # TODO: correct the signal frequencies
    known_sinus_signal = [np.sin(np.arange(0, 1000, 0.1)*np.pi/freq )for freq in range(num_electrodes)]
    known_annotations = [(0, 'Start_1'), (0.5, 'Release_1'), (1, 'Start_2'), (1.5, 'Release_2')]
    monkeypatch.setattr(Recording, 'load_file', lambda: (known_sinus_signal, known_annotations))





class Test_Recording:

    @pytest.mark.slow
    def test_load_file(self, my_file_path_and_exp_name):
        signal, annotations = Recording.load_file(my_file_path_and_exp_name[0])
        assert len(signal) == 19  # len of np.array is the size of dim 0 (16 EMG channels + 3 accelerometer channels)
        assert len(annotations) == 200  # 10 trials for each label (10 labels) = 100 trials * 2 (start and end of trial) = 200

    def test_file_path_to_experiment(self, my_file_path_and_exp_name):
        experiment = Recording.file_path_to_experiment(my_file_path_and_exp_name[0])
        assert experiment == my_file_path_and_exp_name[1]

    def test_match_experiment(self, my_file_path_and_exp_name):
        recording = Recording(my_file_path_and_exp_name[0], Data_Pipeline())
        assert recording.match_experiment(my_file_path_and_exp_name[1])
        assert recording.match_experiment(my_file_path_and_exp_name[2])

    def test_get_dataset_no_synthetics(self, my_file_path_and_exp_name):
        recording = Recording(my_file_path_and_exp_name[0], Data_Pipeline())
        data, labels = recording.get_dataset(include_synthetics = False)
        assert data == recording.features
        assert labels == recording.labels

    def test_get_dataset_with_synthetics(self, my_file_path_and_exp_name):
        recording = Recording(my_file_path_and_exp_name[0], Data_Pipeline())
        data, labels = recording.get_dataset(include_synthetics = True)
        assert data == np.concatenate((recording.features,
                                       recording.extract_features(recording.create_synthetic_subsegments()[0])),
                                       axis = 0)
        assert labels == np.concatenate(recording.labels, recording.create_synthetic_subsegments()[1], axis = 0)

if __name__ == '__main__':
    pytest.main()
