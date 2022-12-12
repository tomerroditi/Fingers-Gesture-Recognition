import os
from pathlib import Path


class Paths_Handler:
    def __init__(self, data_root_path: Path):
        self.root_path = data_root_path
        self.paths = []

    def add_paths_of_subjects_num(self, subjects_num: int or list[int]):
        """This function adds the paths of the subjects to the paths list"""
        if isinstance(subjects_num, int):
            subjects_num = [subjects_num]

        subjects_num = [f'{num:03d}' for num in subjects_num]  # convert to 3 digits string format
        # folders of specified subjects
        folders = [path for path in self.root_path.iterdir() if any([num in path.name for num in subjects_num]) and path.is_dir()]
        for folder in folders:
            sub_folders = [path for path in folder.iterdir() if path.is_dir()]  # the sessions folders
            for sub_folder in sub_folders:
                files = [file for file in sub_folder.iterdir() if file.is_file() and file.suffix == '.edf']  # the edf files
                self.paths.extend(files)
