import mne
from Source.recording import Recording
from Source.pipelines import Data_Pipeline
from Source.data_manager import Data_Manager
from pathlib import Path

pipeline = Data_Pipeline()
dm = Data_Manager([1], pipeline)

# %%
print(dm.data_info())

# %%
dataset = dm.get_dataset(experiments='*_*_*', include_synthetics=False)

# %%
small_dataset = dm.get_dataset(experiments='*_1_1', include_synthetics=False)
