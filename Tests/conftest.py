import pytest
import numpy as np
import mne
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Recording
from pathlib import Path


@pytest.fixture(scope = "session")
def test_dir(tmp_path_factory):
    """This function creates a directory with empty edf files in the same directory structure as our real data"""
    files = [r'subject_001\session_1\GR_pos1_001_S1_Recording_00_SD.edf',
             r'subject_001\session_1\some_file.txt',
             r'subject_001\some_file.mp4',
             r'subject_001\session_2\GR_pos1_001_s2_Recording_00_SD.edf',
             r'subject_001\session_3\GR_pos1_001_S3_Recording_00_SD.edf',
             r'subject_002\session_1\GR_pos1_002_S1_Recording_00_SD.edf',
             r'subject_002\session_1\some_file.idi',
             r'subject_002\session_1\GR_pos2_002_s1_Recording_00_SD.edf',
             r'subject_020\session_1\GR_pos1_020_S1_Recording_00_SD.edf',
             r'subject_005\session_1\GR_pos1_005_S1_Recording_00_SD.edf',
             r'subject_005\session_1\some_file.mp3',
             r'subject_005\session_1\GR_pos3_005_s1_Recording_00_SD.edf']

    subdirs = [r'subject_001\session_1', r'subject_001\session_2', r'subject_001\session_3', r'subject_002\session_1',
                r'subject_020\session_1', r'subject_005\session_1']

    files_path = tmp_path_factory.mktemp('files')
    subdirs = [files_path / subdir for subdir in subdirs]
    [subdir.mkdir(parents = True, exist_ok = True) for subdir in subdirs]
    files = [files_path / file for file in files]

    for file in files:
        path = Path(file)
        with path.open(mode='w') as new_file:
            new_file.close()

    yield files_path


real_file_paths = [
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos1_001_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_004\session_1\GR_pos2_004_S1_Recording_00_SD_edited.edf',
    r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_005\session_1\GR_pos3_005_S1_Recording_00_SD_edited.edf'
            ]
real_file_paths = [Path(path) for path in real_file_paths]


@pytest.fixture(scope = 'session', params = real_file_paths)
def real_file_path(request):
    yield request.param


@pytest.fixture(scope = 'session')
def test_recording_with_loaded_data():
    """This fixture initialize a Recording object with a real file path but loads synthetic data"""
    num_electrodes = 16
    sinus_signal = [np.sin(np.arange(0, 10, 0.00025)*np.pi/(freq*10))for freq in range(num_electrodes)]
    time = np.arange(0, 10, 0.00025)
    annotations = [(2, 'Start_TwoFingers'), (2.5, 'Release_TwoFingers'),
                   (3.2, 'Start_Abduction'), (3.7, 'Release_Abduction'),
                   (4.4, 'Start_ThreeFingers'), (4.9, 'Release_ThreeFingers'),
                   (5.6, 'Start_Bet'), (6.1, 'Release_Bet'),
                   (6.8, 'Start_Bet'), (7.3, 'Release_Bet'),
                   (8, 'Start_Abduction'), (8.5, 'Release_Abduction'),
                   (9.2, 'Start_ThreeFingers'), (9.7, 'Release_ThreeFingers')]
    pipeline = Data_Pipeline()

    rec_for_testing = Recording(real_file_paths[0], pipeline)
    rec_for_testing.raw_edf = mne.io.edf.edf.RawEDF()
    rec_for_testing.annotations = annotations
    rec_for_testing.time = time
    rec_for_testing.signal = sinus_signal
    yield rec_for_testing


paths = [
    r'c:\data_files\subject_001\session_1\GR_pOS1_001_S1_Recording_00_SD_edited.edf',
    r'c:\data_files\subject_004\session_1\Gr_pos2_004_s1_Recording_00_SD_edited.edf',
    r'c:\data_files\subject_005\session_1\gr_pos3_005_S1_Recording_00_SD_edited.edf'
            ]
paths = [Path(path) for path in paths]
exp_names = ['001_1_1', '004_1_2', '005_1_3']
general_exp_names = ['*_1_1', '004_*_*', '005_1_*']

@pytest.fixture(scope = 'session', params = zip(paths, exp_names, general_exp_names))
def file_path_and_exp_name(request):
    yield request.param

