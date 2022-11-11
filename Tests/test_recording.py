import pytest
from Source.recording import Recording

file_paths = [
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos1_001_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos2_001_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos3_001_S1_Recording_00_SD_edited.edf'
    ]
exp_names = ['001_1_1', '001_1_2', '001_1_3']


@pytest.fixture(scope = 'module', params = zip(file_paths, exp_names))
def my_file_path_and_exp_name(request):
    return request.param


class Test_Recording:

    @pytest.mark.slow
    def test_load_file(self, my_file_path_and_exp_name):
        signal, annotations = Recording.load_file(my_file_path_and_exp_name[0])
        assert len(signal) == 19  # len of np.array is the size of dim 0 (16 EMG channels + 3 accelerometer channels)
        assert len(annotations) == 200  # 10 trials for each label (10 labels) = 100 trials * 2 (start and end of trial) = 200

    def test_file_path_to_experiment(self, my_file_path_and_exp_name):
        experiment = Recording.file_path_to_experiment(my_file_path_and_exp_name[0])
        assert experiment == my_file_path_and_exp_name[1]


if __name__ == '__main__':
    pytest.main()
