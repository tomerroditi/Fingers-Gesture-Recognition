from pathlib import Path
import pytest
from Source.paths_handler import Paths_Handler


@pytest.fixture(scope = "module")
def create_folder_with_empty_files(tmp_path_factory):
    """This function creates empty edf files for the class's tests"""
    files = [r'subject_001\session_1\GR_pos1_001_S1_Recording_00_SD.edf',
             r'subject_001\session_2\GR_pos1_001_s2_Recording_00_SD.edf',
             r'subject_001\session_3\GR_pos1_001_S3_Recording_00_SD.edf',
             r'subject_002\session_1\GR_pos1_002_S1_Recording_00_SD.edf',
             r'subject_002\session_1\GR_pos2_002_s1_Recording_00_SD.edf',
             r'subject_020\session_1\GR_pos1_020_S1_Recording_00_SD.edf',
             r'subject_005\session_1\GR_pos1_005_S1_Recording_00_SD.edf',
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


class Test_Paths_handler:
    @pytest.mark.parametrize("subjects, expected", [
        ([1], 3),
        ([1, 2], 5),
        ([1, 2, 20, 5], 8),
        ([8, 9], 0)
    ])
    def test_get_subject_files(self, create_folder_with_empty_files, subjects, expected):
        paths_handler = Paths_Handler(str(create_folder_with_empty_files))
        paths_handler.add_paths_of_subjects_num(subjects)
        num_files = len(paths_handler.paths)
        assert num_files == expected


if __name__ == '__main__':
    pytest.main()
