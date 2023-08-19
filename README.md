# Fingure-Gesture-Recognition
This projects aims to classify finger gestures using the forearm's EMG signals.
The project is divided into 3 main parts:
1. Data Collection
2. Model training
3. Model testing

In this repo you will find 2 packages under the 'Source' directory:
1. fgr - data collection, preprocessing, training and testing modules (written by us)
2. streamer - supports integration with X-trodes DAU real-time streaming (written by X-trodes)

## Data Collection
The data collection process is done using the X-trodes DAU (Data Acquisition Unit) and the X-trodes EMG sensors.
The DAU is connected to the computer via a bluetooth connection (with the help of the X-trodes software), which streams 
the EMG (and in future releases the ACC and GYRO as well) signals to the computer.

### fgr.data_collection
This module is responsible for collecting finger gestures data from the X-trodes DAU.
It contains a single class - Experiment, which is responsible for the data collection process.
In our work we use this class to run experiments for data collections. Read more about the Experiment class in the
class documentation.

## Data Preprocessing
### fgr.pipelines
This module contains a class - Data_Pipeline, which holds the data preprocessing parameters and methods. It is used to 
control the preprocessing process. Read more about the Data_Pipeline class in the class documentation.

### fgr.data_manager
This module is responsible for managing datasets and data preprocessing. 

#### Data_Manager
This class is used to easily extract datasets from specific subjects and experiments.
It is mainly used for offline training and testing. Read more about the Data_Manager class in the class documentation.
It is best suited for working with recordings that were collected with the old X-trodes software (before the bluetooth connection),
mainly due to the files naming convention (so it should be an easy fix in case you want to use it with the new software).

#### Subject
This class is used to gather all the data of a specific subject. It is used as a helper class for the Data_Manager class.

#### Recording_Preprocessing
A class that holds common preprocessing functions to be used on experiences recordings data.
We gathered these function into a seperated class, so we could make sure that the same preprocessing functions are activated
on different recordings, which will be elaborated on later.

#### Recording_Emg_Live
Used for live recordings with EMG data only (getting data from local variables).

#### Recording_Emg
Used for offline data recordings with EMG data only (getting data from saved files).

#### Recording_Emg_Acc
Used for offline data recordings with EMG and ACC data (getting data from saved files).

#### Real_Time_Recording
Used for live predictions with EMG data only (getting data from a streamer.data.Data obj).


## Model Training

### fgr.models
This module contains classes for training and testing models, and for evaluating the results.
It also has a class for real time predictions.

#### Real_Time_Predictor
This class is used for real time predictions. it is utilizing the Real_Time_Recording class from the data_manager module
to do the preprocessing of the data.

#### Simple_CNN
a base class that holds training and evaluation procedures for a simple CNN model.
read the class documentation for more information.

#### Net
Our main model class. It is a subclass of the Simple_CNN class, and it holds the model architecture.




