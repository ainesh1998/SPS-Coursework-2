#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'
class_colours = [CLASS_1_C,CLASS_2_C,CLASS_3_C]

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

#################################

#     Helper Functions     #

def getColours(index,train_set):
    onlyColours = []
    for i in range(len(train_labels)):
        if(train_labels[i] == index + 1):
            onlyColours.append(train_set[i])
    arr = np.array(onlyColours)
    return arr

# Calculates Euclidean distance
def distance(x, y):
    result = 0
    for i in range(x.size):
        result += (x[i] - y[i])*(x[i] - y[i])
    return result

# Calculates the accuracy of a classifier
def calculate_accuracy(gt_labels, pred_labels):
    count = 0
    for i in range(gt_labels.size):
        if gt_labels[i] == pred_labels[i]:
            count += 1

    return count/gt_labels.size

################################

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of
    # the function
    # scatter plot bit
    n_features = train_set.shape[1]
    fig, ax = plt.subplots((n_features), (n_features))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    for i in range(len(class_colours)):
        for j in range(n_features):
            for k in range(n_features):
                arr = getColours(i,train_set)
                if(len(arr) != 0):
                    ax1.scatter(arr[:,j],arr[:,k],c = class_colours[i])
                    ax[j,k].xaxis.set_visible(False)
                    ax[j,k].yaxis.set_visible(False)
                    ax[j,k].scatter(arr[:,j],arr[:,k],c = class_colours[i])
    plt.show()
    return [12,10]


def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    # nearest centroid bit

    # reduce the train set to the features we selected
    selected_features = [10,12]
    reduced_train_set = train_set[:, selected_features[0]]

    for feature in range(1, len(selected_features)):
        reduced_train_set = np.column_stack((reduced_train_set, train_set[:, selected_features[feature]]))

    classifiedTests = [] # stores the result of classification for each data sample

    for test in test_set:
        closestNeighbourIndices = [] # stores the indices of the nearest neighbours found so far
        closestNeighbourClasses = [] # stores the classes of the nearest neighbours found so far

        # find the k nearest neighbours for the test sample
        for i in range(k):
            minDist = np.Infinity
            nearestClass = 0
            nearestPointIndex = 0
            for train in range(len(train_set)):
                if distance(test, train_set[train]) < minDist and train not in closestNeighbourIndices:
                    minDist = distance(test, train_set[train])
                    nearestClass = train_labels[train]
                    nearestPointIndex = train
            closestNeighbourClasses.append(nearestClass)
            closestNeighbourIndices.append(nearestPointIndex)

        # find the most common class among the neighbours (found this online)
        majorityClass = max(set(closestNeighbourClasses), key = closestNeighbourClasses.count)
        classifiedTests.append(majorityClass)

    return classifiedTests


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
