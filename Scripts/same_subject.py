from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

# %% an example of how to use the data manager
my_subject = 1
data_pipeline = Data_Pipeline()
database = Data_Manager([1, 2, 3, 4, 5, 6], data_pipeline)
train_data, train_labels = database.get_dataset([f'{my_subject:03}_1_1', f'{my_subject:03}_1_2'])
test_data, test_labels = database.get_dataset([f'{my_subject:03}_1_3'])


