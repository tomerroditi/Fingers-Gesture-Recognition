from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager
import Source.fgr.models as models
from warnings import simplefilter
from pathlib import Path
import torch

# %% pipeline definition and data manager creation
subject_num = 1
data_path = Path(r'..\data')
pipeline = Data_Pipeline(base_data_files_path=data_path, emg_sample_rate=250, emg_low_freq=35, emg_high_freq=124,
                         features_norm='none')  # configure the data pipeline you would like to use (check pipelines module for more info)
dm = Data_Manager([subject_num], pipeline)
print(dm.data_info())

# %% extract datasets from the data manager
dataset_train = dm.get_dataset(experiments=f'{subject_num:03d}_*_*')
# rec = dm.subjects[0].recordings[0]
# rec.heatmap_visualization(5)

data_train = dataset_train[0]
data_train = data_train.reshape(data_train.shape[0], 1, 4, 4)  # reshape to fit the CNN input
labels_train = dataset_train[1]

# %% set and train a model (cv or not)
model = models.Net(num_classes=10, dropout_rate=0.1)
model.fit_model(data_train, labels_train, num_epochs=20, batch_size=64, lr=0.001, l2_weight=0.0001)
model.evaluate_model(model.train_data, model.train_labels, cm_title='model results')
torch.save(model, f'..\\data\\{subject_num:03d}\\model.pth')

# models, accuracies = model.cv_fit_model(data, labels, num_epochs=200, batch_size=64, lr=0.001, l2_weight=0.0001)
# # models evaluation
# print(f'model average accuracy: {np.mean(accuracies)}')
# for i, model in enumerate(models):
#     model.evaluate_model(model.test_data, model.test_labels, cm_title='model number ' + str(i))
#
