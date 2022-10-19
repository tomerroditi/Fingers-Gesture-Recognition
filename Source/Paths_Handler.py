class Paths_Handler:
    def __init__(self, data_root_path: str):
        self.root_path = data_root_path
        self.paths = []

    @property
    def paths(self):
        return self._paths

    @paths.setter
    def paths(self, paths):
        self._paths = paths

    def add_paths_of(self, positions: {list, int}, subjects: {list, int}, sessions: {list, int}):
        args = [positions, subjects, sessions]
        is_list = [isinstance(arg, list) for arg in args]
        if any(is_list):
            if not all(is_list) or not len(subjects) == len(positions) == len(sessions):
                raise PathConstructionError('''you must supply evenly length lists of subjects, positions and sessions!
                                               You may as well use a single int value for each argument''')

            for i in range(len(subjects)):
                path = self.make_path(positions[i], subjects[i], sessions[i])
                self.paths.append(path)
        else:
            path = self.make_path(positions, subjects, sessions)
            self.paths.append(path)

    @staticmethod
    def make_path(position: int, subject: int, session: int):
        args = [position, subject, session]
        if all([isinstance(arg, int) for arg in args]):
            return f'GR_pos{position}_{subject:0>3d}_S{session}_Recording_00_SD.edf'
        else:
            raise PathConstructionError('you must supply int values for all input arguments')


class PathConstructionError(Exception):
    pass
