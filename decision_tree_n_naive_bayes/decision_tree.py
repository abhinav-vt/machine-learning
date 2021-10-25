"""This module includes methods for training and predicting using decision trees."""
import math

import numpy as np


def calculate_information_gain(data, labels):
    """
    Computes the information gain for each feature in data

    :param data: d x n matrix of d features and n examples
    :type data: ndarray
    :param labels: n x 1 vector of class labels for n examples
    :type labels: array
    :return: d x 1 vector of information gain for each feature
    :rtype: array
    """
    all_labels = np.unique(labels)
    num_classes = len(all_labels)

    class_count = np.zeros(num_classes)

    d, n = data.shape

    parent_entropy = 0
    for c in range(num_classes):
        class_count[c] = np.sum(labels == all_labels[c])
        if class_count[c] > 0:
            class_prob = class_count[c] / n
            parent_entropy -= class_prob * np.log(class_prob)

    # print("Parent entropy is %d\n" % parent_entropy)

    gain = parent_entropy * np.ones(d) #initialization of gains for every attribute

    # we use a matrix dot product to sum to make it more compatible with sparse matrices
    num_x = data.dot(np.ones(n)) # number of examples containing each feature
    prob_x = num_x / n # fraction of examples containing each feature
    prob_not_x = 1 - prob_x

    for c in range(num_classes):
        # print("Computing contribution of class %d." % c)
        num_y = np.sum(labels == all_labels[c])
        # this next line sums across the rows of data, multiplied by the
        # indicator of whether each column's label is c. It counts the number
        # of times each feature is on among examples with label c.
        # We again use the dot product for sparse-matrix compatibility
        data_with_label = data[:, labels == all_labels[c]]
        num_y_and_x = data_with_label.dot(np.ones(data_with_label.shape[1]))

        # Prevents Python from outputting a divide-by-zero warning
        with np.errstate(invalid='ignore'):
            prob_y_given_x = num_y_and_x / (num_x + 1e-8) # probability of observing class c for each feature
        prob_y_given_x[num_x == 0] = 0

        nonzero_entries = prob_y_given_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_x, prob_y_given_x), np.log(prob_y_given_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

        # The next lines compute the probability of y being c given x = 0 by
        # subtracting the quantities we've already counted
        # num_y - num_y_and_x is the number of examples with label y that
        # don't have each feature, and n - num_x is the number of examples
        # that don't have each feature
        with np.errstate(invalid='ignore'):
            prob_y_given_not_x = (num_y - num_y_and_x) / ((n - num_x) + 1e-8)
        prob_y_given_not_x[n - num_x == 0] = 0

        nonzero_entries = prob_y_given_not_x > 0
        if np.any(nonzero_entries):
            with np.errstate(invalid='ignore', divide='ignore'):
                children_entropy = - np.multiply(np.multiply(prob_not_x, prob_y_given_not_x), np.log(prob_y_given_not_x))
            gain[nonzero_entries] -= children_entropy[nonzero_entries]

    return gain


def decision_tree_train(train_data, train_labels, params):
    """Train a decision tree to classify data using the entropy decision criterion.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. Must include a 'max_depth' value
    :type params: dict
    :return: dictionary encoding the learned decision tree
    :rtype: dict
    """
    max_depth = params['max_depth']

    labels, counts = np.unique(train_labels, return_counts=True)

    num_classes = labels.size

    model = recursive_tree_train(train_data, train_labels, depth=0, max_depth=max_depth, num_classes=num_classes)

    return model


def recursive_tree_train(data, labels, depth, max_depth, num_classes):
    """Helper function to recursively build a decision tree by splitting the data by a feature.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param labels: length n numpy array with integer labels
    :type labels: array_like
    :param depth: current depth of the decision tree node being constructed
    :type depth: int
    :param max_depth: maximum depth to expand the decision tree to
    :type max_depth: int
    :param num_classes: number of classes in the classification problem
    :type num_classes: int
    :return: dictionary encoding the learned decision tree node
    :rtype: dict
    """

    # find all class labels and their respective counts
    unique, counts = np.unique(labels, return_counts=True)

    # recursion base case: set prediction value as the class with highest count
    # when either the max_depth is reached or if all labels are same
    if depth >= max_depth or len(unique) == 1:
        return {'prediction': unique[np.argmax(counts)]}

    # find the splitting feature based on the one with highest information gain
    info_gain = calculate_information_gain(data=data, labels=labels)
    split_index = np.argmax(np.array(info_gain))

    # assign all data with splitting feature with True value to left sub tree
    # and vice versa for False values
    left_split_indices = np.where(data[split_index] == True)[0]
    right_split_indices = np.where(data[split_index] == False)[0]

    node = {}
    if len(left_split_indices) > 0:
        # build left sub tree after removing the current splitting feature
        node['left'] = recursive_tree_train(data=np.delete(data[:, left_split_indices], split_index, 0),
                                            labels=labels[left_split_indices],
                                            depth=depth + 1, max_depth=max_depth, num_classes=num_classes)
    else:
        # if left indices size is 0 then set left tree as leaf node with prediction as the class with highest count
        # and vice versa for right
        node['left'] = {'prediction': unique[np.argmax(counts)]}

    if len(right_split_indices) > 0:
        node['right'] = recursive_tree_train(data=np.delete(data[:, right_split_indices], split_index, 0),
                                             labels=labels[right_split_indices],
                                             depth=depth + 1, max_depth=max_depth, num_classes=num_classes)
    else:
        node['right'] = {'prediction': unique[np.argmax(counts)]}

    # save the splitting feature as a node attribute - to be used for prediction
    node['feature_idx'] = split_index
    return node


def prediction_util(record, model):
    """
    Helper function for traversing through the decision tree to find predicted class
    :param record: d X 1 data row for which to find the predicted label
    :type record: ndarray
    :param model: decision tree found after training
    :type model: dict
    :return: predicted class
    :rtype: int
    """

    # recursion base case: if 'prediction' key exists return value
    if model.get('prediction', None) is not None:
        return model.get('prediction')

    # get the splitting feature's index
    feature_idx = model['feature_idx']

    # if value at split feature is True: recurse on left sub-tree
    # else recurse on right sub-tree
    if record[feature_idx]:
        return prediction_util(record=np.delete(record, feature_idx), model=model['left'])
    return prediction_util(record=np.delete(record, feature_idx), model=model['right'])


def decision_tree_predict(data, model):
    """Predict most likely label given computed decision tree in model.

    :param data: d x n ndarray of d binary features for n examples.
    :type data: ndarray
    :param model: learned decision tree model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """

    n = len(data[0])
    predictions = np.zeros(n)
    for i in range(n):
        predictions[i] = prediction_util(data[:, i], model)

    return predictions

