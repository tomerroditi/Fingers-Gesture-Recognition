import pytest
import numpy as np
from Source.fgr.data_manager import Recording
from Source.fgr.pipelines import Data_Pipeline


class Test_Recording:

    @pytest.mark.slow
    def test_load_file(self, real_file_path):
        rec = Recording(real_file_path, Data_Pipeline())
        rec.load_file()
        assert np.shape(rec.signal)[1] == np.shape(rec.time)[0]
        assert np.shape(rec.signal)[0] == 19, "len of np.array is the size of dim 0 (16 EMG channels + 3 accelerometer channels)"
        assert len(rec.annotations) == 200, "10 trials for each label (10 labels) = 100 trials * 2 (start and end of trial) = 200"

    def test_file_path_to_experiment(self, file_path_and_exp_name):
        experiment = Recording.file_path_to_experiment(file_path_and_exp_name[0])
        assert experiment == file_path_and_exp_name[1], "experiment name is not correct"

    def test_match_experiment(self, file_path_and_exp_name):
        recording = Recording(file_path_and_exp_name[0], Data_Pipeline())
        assert recording.match_experiment(file_path_and_exp_name[1]) is True, "should be an exact match"
        assert recording.match_experiment(file_path_and_exp_name[2]) is True, "should be a general match"

    @pytest.mark.xfail
    def test_get_dataset_with_synthetics(self, test_recording_with_loaded_data):
        rec = test_recording_with_loaded_data
        data, labels = rec.get_dataset(include_synthetics = True)
        assert np.shape(data)[0] > np.shape(rec.features)[0], "should have more data points than the original recording"
        assert len(labels) > len(rec.labels), "should have more labels than the original recording"


if __name__ == '__main__':
    pytest.main()
