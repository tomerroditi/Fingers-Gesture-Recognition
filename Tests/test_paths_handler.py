from pathlib import Path
import pytest
from Source.paths_handler import Paths_Handler


class Test_Paths_handler:
    @pytest.mark.parametrize("subjects, expected", [
        ([1], 3),
        ([1, 2], 5),
        ([1, 2, 20, 5], 8),
        ([8, 9], 0)
    ])
    def test_get_subject_files(self, test_dir, subjects, expected):
        paths_handler = Paths_Handler(test_dir)
        paths_handler.add_paths_of_subjects_num(subjects)
        num_files = len(paths_handler.paths)
        assert num_files == expected


if __name__ == '__main__':
    pytest.main()
