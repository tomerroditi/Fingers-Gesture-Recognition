from pathlib import Path
import pytest
from Source.Paths_Handler import Paths_Handler


@pytest.fixture(scope = "module")
def create_folder_with_empty_files(tmp_path_factory):
    """This function creates empty edf files for the class's tests"""
    files = ['GR_pos1_001_S1_Recording_00_SD.edf',
             'GR_pos1_001_s2_Recording_00_SD.edf',
             'GR_pos1_001_S3_Recording_00_SD.edf',
             'GR_pos1_002_S1_Recording_00_SD.edf',
             'GR_pos2_002_s1_Recording_00_SD.edf',
             'GR_pos1_020_S1_Recording_00_SD.edf',
             'GR_pos1_005_S1_Recording_00_SD.edf',
             'GR_pos3_005_s1_Recording_00_SD.edf']

    files_path = tmp_path_factory.mktemp('files')
    for file in files:
        path = Path(files_path, file)
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
        paths_handler.add_paths_of_subjects(subjects)
        num_files = len(paths_handler.paths)
        assert num_files == expected


if __name__ == '__main__':
    pytest.main()
