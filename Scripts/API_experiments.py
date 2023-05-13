from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager
import Source.fgr.models as models
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# %% pipeline definition and data manager creation
pipeline = Data_Pipeline()  # configure the data pipeline you would like to use (check pipelines module for more info)
dm = Data_Manager([18], pipeline)
print(dm.data_info())

# %% extract datasets from the data manager
dataset_train = dm.get_dataset(experiments='018_*_4')
# dataset_test = dm.get_dataset(experiments='018_2_*')

data_train = dataset_train[0]
# data_test = dataset_test[0]

# extract labels - labels includes the gesture number and experiment name as well, overall the labels are in the
# format of "<subject>_<session>_<position>_<gesture>_<iteration_number>"
labels_train = dataset_train[1]
# labels_test = dataset_test[1]

# reshape the data to match the model architecture
data_train = data_train.reshape(data_train.shape[0], 1, 4, 4)  # reshape to fit the CNN input
# data_test = data_test.reshape(data_test.shape[0], 1, 4, 4)  # reshape to fit the CNN input

# %% set and train a model (cv or not)
model = models.Net(num_classes=10, dropout_rate=0.3)
model.fit_model(data_train, labels_train, num_epochs=200, batch_size=64, lr=0.001, l2_weight=0.0001)
                # test_data=data_test, test_labels=labels_test)
model.evaluate_model(model.val_data, model.val_labels, cm_title='model results')

# models, accuracies = model.cv_fit_model(data, labels, num_epochs=200, batch_size=64, lr=0.001, l2_weight=0.0001)
# # models evaluation
# print(f'model average accuracy: {np.mean(accuracies)}')
# for i, model in enumerate(models):
#     model.evaluate_model(model.test_data, model.test_labels, cm_title='model number ' + str(i))
#
