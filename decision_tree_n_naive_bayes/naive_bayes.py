"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    labels, counts = np.unique(train_labels, return_counts=True)

    d, n = train_data.shape
    num_classes = labels.size

    # TODO: INSERT YOUR CODE HERE TO LEARN THE PARAMETERS FOR NAIVE BAYES (USING LAPLACE SMOOTHING)

    model = {}

    # partition training data by each class
    X_by_class = [train_data[:, train_labels == c] for c in labels]

    # take log of probabality from counts of each class by number of training example
    # also added laplace smoothening
    model['prior'] = np.log([(np.shape(X_class)[1] + 1) / (n + num_classes) for X_class in X_by_class])

    # find counts of each feature for each class
    counts_by_class = np.array([np.sum(x, axis=1) for x in X_by_class])

    # find conditional probability values for True features
    conditional = (counts_by_class + 1) / (counts[:, None] + 2)
    # find complement of above conditional probability value for False features
    conditional_complement = 1 - conditional

    # take log of probabilities for easy dot product while prediction
    model['conditional'] = np.log(conditional)
    model['conditional_complement'] = np.log(conditional_complement)
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    # TODO: INSERT YOUR CODE HERE FOR USING THE LEARNED NAIVE BAYES PARAMETERS
    # TO CLASSIFY THE DATA

    # extract prior and conditional probabilities from model
    prior, conditional, conditional_complement = model['prior'], model['conditional'], model['conditional_complement']

    # take dot product of conditional log probability with given data for features with value True
    prob = conditional.dot(data)

    # add dot product of conditional_complement log probability with given data
    # for features with value False ( after inverting all features)
    prob += conditional_complement.dot(np.logical_not(data))

    # add the prior log probability
    net_prob = np.transpose(prob) + prior

    # determine prediction based on class with highest probability for each data point
    result = np.argmax(net_prob, axis=1)

    return result
