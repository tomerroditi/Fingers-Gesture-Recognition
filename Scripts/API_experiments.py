import numpy as np
from Source.pipelines import Data_Pipeline
from Source.data_manager import Data_Manager
import Source.models as models
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# %% pipeline definition and data manager creation
pipeline = Data_Pipeline()
dm = Data_Manager([1], pipeline)
print(dm.data_info())

# %% extract datasets from the data manager
dataset = dm.get_dataset(experiments='001_*_2')
data = dataset[0]
labels = dataset[1]  # labels includes the gesture number and experiment name as well

# reshape the data to match the model architecture
data = data.reshape(data.shape[0], 1, 4, 4)  # reshape to fit the CNN input

# set and train a model
model = models.Net(num_classes=10, dropout_rate=0.3)
models, accuracies = model.cv_fit_model(data, labels, num_epochs=200, batch_size=64, lr=0.001, l2_weight=0.0001)
# models evaluation
print(f'model average accuracy: {np.mean(accuracies)}')
for i, model in enumerate(models):
    model.evaluate_model(model.test_data, model.test_labels, cm_title='model number ' + str(i))

