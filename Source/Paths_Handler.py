import os


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

    def add_paths_of_subjects_num(self, subjects_num: {int, list}):
        """This function adds the paths of the subjects to the paths list"""
        subjects_num = list(subjects_num)
        subjects_num = [f'{num:03d}' for num in subjects_num]
        folders = os.listdir(self.root_path)
        selectors = [any(num in folder for num in subjects_num) for folder in folders]
        folders = [folder for folder, selector in zip(folders, selectors) if selector]  # folders of specified subjects
        for folder in folders:
            sub_folders = os.listdir(os.path.join(self.root_path, folder))
            for sub_folder in sub_folders:
                files = os.listdir(os.path.join(self.root_path, folder, sub_folder))
                for file in files:
                    self.paths.append(os.path.join(self.root_path, folder, sub_folder, file))
