"""This module includes utilities to run cross-validation on general supervised learning methods."""
from __future__ import division
import numpy as np


def cross_validate(trainer, predictor, all_data, all_labels, folds, params):
    """Perform cross validation with random splits.

    :param trainer: function that trains a model from data with the template
             model = function(all_data, all_labels, params)
    :type trainer: function
    :param predictor: function that predicts a label from a single data point
                label = function(data, model)
    :type predictor: function
    :param all_data: d x n data matrix
    :type all_data: numpy ndarray
    :param all_labels: n x 1 label vector
    :type all_labels: numpy array
    :param folds: number of folds to run of validation
    :type folds: int
    :param params: auxiliary variables for training algorithm (e.g., regularization parameters)
    :type params: dict
    :return: tuple containing the average score and the learned models from each fold
    :rtype: tuple
    """
    scores = np.zeros(folds)

    d, n = all_data.shape

    indices = np.array(range(n), dtype=int)

    # pad indices to make it divide evenly by folds
    examples_per_fold = int(np.ceil(n / folds))
    ideal_length = int(examples_per_fold * folds)
    # use -1 as an indicator of an invalid index
    indices = np.append(indices, -np.ones(ideal_length - indices.size, dtype=int))
    assert indices.size == ideal_length

    indices = indices.reshape((examples_per_fold, folds))

    models = []

    for i in range(folds):
        # delete the i^th fold from training data
        training_indices = np.delete(indices, i, 1).flatten()
        # exclude all indices with -1 value
        training_indices = training_indices[training_indices != -1]

        # setting i^th fold as testing indices
        testing_indices = indices[:, i]

        # get training and testing data
        train_data = all_data[:, training_indices]
        test_data = all_data[:, testing_indices]

        # get train and testing labels
        train_labels = all_labels[training_indices]
        test_labels = all_labels[testing_indices]

        # train model
        model = trainer(train_data, train_labels, params)

        models.append(model)

        # evaluate model for test data
        predictions = predictor(test_data, model)

        # get mean accuracy for i^th fold
        scores[i] = np.mean(predictions == test_labels)

    return np.mean(scores), models
