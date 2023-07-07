import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_by_gesture(*arrays: np.array, labels: np.array = None, test_size: float = 0.2, seed: int = 42)\
        -> (np.array, np.array, np.array, np.array) or (list[np.array], list[np.array], np.array, np.array):
    """
    split data into train test sets, each set will contain data from different gestures repetitions, e.g. if we have
    10 repetitions of "fist" gestures from an experiment ('001_1_1_fist_1', '001_1_1_fist_2', ..., '001_1_1_fist_10') or
    ('fist_001_1_1_1_1', 'fist_001_1_1_1_2', ..., 'fist_001_1_1_1_10').
    then the fist 8 repetitions will be in the train set and the last 2 will be in the test set.
    inputs:
        arrays: numpy arrays, each array should have the same number of rows (samples) as the labels array
        labels: numpy array of labels, each label should be a string, their format is one of both:
                'gesture_subject_position_session_trial_repetition' - new
                'subject_position_session_gesture_repetition' - old
        test_size: float, the percentage of the data to be used for testing
    outputs:
        train_arrays: list of numpy arrays, each array contains the train data (in the same order as the input
                      arrays), if only one array was given as input then a single numpy array will be returned
        test_arrays: list of numpy arrays, each array contains the test data (in the same order as the input
                     arrays), if only one array was given as input then a single numpy array will be returned
        train_labels: numpy array, contains the train labels
        test_labels: numpy array, contains the test labels
    """
    unique_labels = np.unique(labels)
    # remove only the last '<num>' from each label, it is the gesture number
    unique_labels_no_num = np.char.rstrip(unique_labels, '0123456789')

    # in case there is a gesture with one repetition we will discard it since we cant split it to train and test
    only_one_rep = np.array([np.sum(unique_labels_no_num == label) == 1 for label in unique_labels_no_num])
    print(f'number of gestures with only one repetition: {np.sum(only_one_rep)}\n'
          f'Their names are: {unique_labels_no_num[only_one_rep]}')
    unique_labels = unique_labels[np.logical_not(only_one_rep)]
    unique_labels_no_num = unique_labels_no_num[np.logical_not(only_one_rep)]

    # split each gesture of each subject to train and test
    train_gestures, test_gestures = train_test_split(unique_labels, stratify=unique_labels_no_num, test_size=test_size,
                                                     random_state=seed)

    # now split the data itself to train and test
    train_arrays = []
    test_arrays = []
    for array in arrays:
        train_arrays.append(array[np.isin(labels, train_gestures)])
        test_arrays.append(array[np.isin(labels, test_gestures)])

    # split the labels to train and test
    train_labels = labels[np.isin(labels, train_gestures)]
    test_labels = labels[np.isin(labels, test_gestures)]

    # in case we have more than one array we will return a list of arrays, otherwise we will return a single array
    if len(arrays) == 1:
        return train_arrays[0], test_arrays[0], train_labels, test_labels
    else:
        return train_arrays, test_arrays, train_labels, test_labels

