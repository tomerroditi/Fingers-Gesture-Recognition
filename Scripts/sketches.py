from pathlib import Path
from Source.recording import Recording
from Source.pipelines import Data_Pipeline
#%%
paths = [Path(r'G:\.shortcut-targets-by-id\1KV37VQD97wDVcG9cls4l_H4HSiE3GPkI\data_files\subject_001\session_1\GR_pos1_001_S1_Recording_00_SD_edited.edf')]

pipeline = Data_Pipeline()  # default values pipeline
rec = Recording(paths, pipeline)
rec.load_file()

# %%


