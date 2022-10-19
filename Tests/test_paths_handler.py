import pytest
from Source.Paths_Handler import Paths_Handler


class Test_Paths_handler:
    root_path = ''

    @pytest.mark.parametrize("position, subject, session, expected", [
        (1, 1, 1, 'GR_pos1_001_S1_Recording_00_SD.edf'),
        (0, 0, 0, 'GR_pos0_000_S0_Recording_00_SD.edf'),
        (0, 1000, 13, 'GR_pos0_1000_S13_Recording_00_SD.edf'),
    ])
    def test_make_path_normal_use(self, position, subject, session, expected):
        path = Paths_Handler.make_path(position, subject, session)
        assert path == expected

    @pytest.mark.parametrize("position, subject, session, expected", [
        ([], 1, 1, 'PathConstructionError'),
        (0, {}, 0, 'PathConstructionError'),
        ('1', 4, 1, 'PathConstructionError'),
        (10, 0, float('nan'), 'PathConstructionError'),
        (5, (), 17, 'PathConstructionError'),
    ])
    def test_make_path_empty_input(self, position, subject, session, expected):
        with pytest.raises(Exception) as e_info:
            path = Paths_Handler.make_path(position, subject, session)
        assert e_info.typename == 'PathConstructionError'

    @pytest.mark.parametrize('positions, subjects, sessions, expected', [
        ([1, 2, 3, 9], [1, 1, 1, 50], [2, 3, 4, 10], 4),
        ([1, 2, 3], [1, 1, 1], [2, 3, 4], 3),
        ([1], [2], [3], 1),
        (1, 1, 1, 1)
    ])
    def test_add_paths_of_with_valid_input(self, positions, subjects, sessions, expected):
        paths_handler = Paths_Handler(self.root_path)
        paths_handler.add_paths_of(positions, subjects, sessions)
        num_paths = len(paths_handler.paths)
        assert num_paths == expected

    @pytest.mark.parametrize('positions, subjects, sessions, expected', [
        ([1, 2, 3], [1, 1, 1, 50], [2, 3, 4, 10], 'PathConstructionError'),
        (1, [1, 1], [2, 4], 'PathConstructionError'),
        ([1], 2, 3, 'PathConstructionError'),
        ([], 1, 1, 'PathConstructionError')
    ])
    def test_add_paths_of_exceptions(self, positions, subjects, sessions, expected):
        paths_handler = Paths_Handler(self.root_path)
        with pytest.raises(Exception) as e_info:
            paths_handler.add_paths_of(positions, subjects, sessions)
        assert e_info.typename == expected


if __name__ == '__main__':
    pytest.main()
