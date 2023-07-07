import numpy as np
import pytest
from Source.fgr.utils import train_test_split_by_gesture


@pytest.fixture(scope='module')
def dummy_data():
    labels = ['fist_001_1_1', 'two_001_1_1', 'three_001_1_1']
    labels = [label + f'_{i}' for label in labels for i in range(10)]
    arrays = [np.random.rand(1, 4, 4) for _ in range(len(labels))]
    yield np.array(arrays), np.array(labels)


def test_basic_functionality(dummy_data):
    data, labels = dummy_data
    train_arrays, test_arrays, train_labels, test_labels = train_test_split_by_gesture(data, labels=labels, test_size=0.5, seed=42)

    # Assert that the correct data types are returned
    assert isinstance(train_arrays, np.ndarray)
    assert isinstance(test_arrays, np.ndarray)
    assert isinstance(train_labels, np.ndarray)
    assert isinstance(test_labels, np.ndarray)

    # Assert the data is split correctly
    assert len(train_arrays) == len(train_labels)
    assert len(test_arrays) == len(test_labels)
    assert (set(train_labels) & set(test_labels)) == set()


def test_multi_array_input(dummy_data):
    data, labels = dummy_data
    data = [data, data]

    train_array, test_array, train_labels, test_labels = train_test_split_by_gesture(*data, labels=labels,
                                                                                     test_size=0.5, seed=42)

    # Assert that the correct data types are returned
    assert isinstance(train_array, list)
    assert isinstance(test_array, list)
    assert isinstance(train_labels, np.ndarray)
    assert isinstance(test_labels, np.ndarray)


def test_invalid_test_size(dummy_data):
    data, labels = dummy_data
    with pytest.raises(ValueError):
        train_test_split_by_gesture(data, labels=labels, test_size=1.5, seed=42)

