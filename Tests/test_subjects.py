import pytest
import numpy as np
from pathlib import Path
from Source.fgr.data_manager import Recording
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Subject

subjects_num = [1, 2, 5, 20, 30]
num_files = [3, 2, 2, 1]


class Test_Subject:
    @pytest.mark.parametrize("subject_num, n_files", zip(subjects_num, num_files))
    def test_my_files(self, monkeypatch, test_dir, subject_num, n_files):
        pipeline = Data_Pipeline(base_data_files_path=test_dir)
        my_subject = Subject(subject_num, pipeline)
        files = my_subject.my_files()
        assert len(files) == n_files

    @pytest.mark.slow
    def test_load_recordings(self):
        my_subject = Subject(1, Data_Pipeline())
        my_subject.load_recordings()
        assert my_subject.recordings[0].signal.size != 0

    @pytest.mark.parametrize("experiments, expected", [('*_*_*', ['001_1_1', '001_2_1', '001_3_1']),
                                                        ('*_1_*', ['001_1_1']),
                                                        ('*_2_*', ['001_2_1']),
                                                        ('*_*_1', ['001_1_1', '001_2_1', '001_3_1']),
                                                        ('*_*_2', []),
                                                        ('*_1_1', ['001_1_1']),
                                                        ('*_2_1', ['001_2_1']),
                                                        ('2_*_*', [])])
    def test_get_my_experiments(self, test_dir, experiments, expected):
        pipeline = Data_Pipeline(base_data_files_path=test_dir)
        my_subject = Subject(1, pipeline)  # note that this refers to subject num 1 in the test directory (conftest.py)
        my_subject.load_recordings(load_data=False)
        my_experiments = my_subject.get_my_experiments(experiments)
        assert len(my_experiments) == len(expected)
        assert sorted(my_experiments) == sorted(expected)

    def test_get_dataset_without_synthetics(self, monkeypatch):
        my_subject = Subject(1, Data_Pipeline())
        num_rec_in_subject = 3
        rec = Recording(Path(r'c:\001_pos1_S1'), Data_Pipeline())
        rec_feat = np.random.rand(3, 4, 4)
        rec_labels = ['a', 'b', 'c']
        monkeypatch.setattr(rec, 'get_dataset', lambda x: (rec_feat, rec_labels))
        my_subject.recordings = [rec for i in range(num_rec_in_subject)]
        data, labels = my_subject.get_datasets(experiments = '*_*_*')
        assert np.array_equal(data, np.concatenate([rec_feat for i in range(num_rec_in_subject)], axis=0))
        assert np.array_equal(labels, np.concatenate([rec_labels for i in range(num_rec_in_subject)], axis=0))

    @pytest.mark.parametrize("subject, expected", [
        (1, 6),
        (2, 3),
        (19, 1),
    ])
    def test_my_files(self, subject, expected):
        my_subject = Subject(subject, Data_Pipeline())
        paths = my_subject.my_files()
        assert len(paths) == expected

