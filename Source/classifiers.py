from torch.utils.data import dataset
import torch.nn as nn
from .pipelines import Model_Pipeline
from sklearn.svm import SVC


class Classifier_Factory:
    def __init__(self, model_pipeline: Model_Pipeline, train_set: dataset, test_set: dataset):
        self.model_pipeline = model_pipeline
        self.train_set = train_set
        self.test_set = test_set

    @staticmethod
    def get_classifiers_names() -> list[str]:
        return ['svm', 'nn']

    def get_classifier(self, classifier_name: str, **kwargs) -> nn | SVC:
        """
        get a classifier from the factory
        """
        if classifier_name == 'svm':
            return self.my_svm(**kwargs)
        elif classifier_name == 'nn':
            return NN(self.model_pipeline, **kwargs)
        else:
            raise ValueError(f'Classifier {classifier_name} is not supported')

    @staticmethod
    def my_svm(**kwargs):
        return SVC(**kwargs)