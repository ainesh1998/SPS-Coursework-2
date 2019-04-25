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
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.decomposition import PCA
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

def calculate_confusion_matrix(gt_labels, pred_labels,classes):
    confusion_mat = []
    for label in classes : # label = class i
        row = []
        for pred in classes: # pred = class j
            total_count = 0
            pred_count = 0
            for i in range(len(gt_labels)):
                if(gt_labels[i] == label):
                    total_count += 1 #no of samples belonging to class i
                    if(pred_labels[i] == pred):# no of samples belonging to class i that were classified as class j
                        pred_count += 1
            val = pred_count/total_count #cell of matrix
            row.append(val)
        confusion_mat.append(row)
    confusion_mat = np.row_stack(confusion_mat)
    return confusion_mat
def plot_matrix(matrix, ax=None):
    if ax is None:
        ax = plt.gca()
    plt.imshow(matrix,cmap=plt.get_cmap('summer'))
    plt.colorbar()
    posY = 0
    for i in range(len(matrix)):
        posX = 0
        for j in range(len(matrix[0])):
            text = str(round(matrix[i,j],3))
            plt.text(posX,posY,text)
            posX += 1
        posY += 1
    plt.show()
################################

def feature_selection(train_set, train_labels, **kwargs):
    # write your code here and make sure you return the features at the end of
    # the function
    # scatter plot bit
    # n_features = train_set.shape[1]
    # fig, ax = plt.subplots((n_features), (n_features))
    # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
    # for i in range(len(class_colours)):
    #     for j in range(n_features):
    #         for k in range(n_features):
    #             arr = getColours(i,train_set)
    #             if(len(arr) != 0):
    #                 ax[j,k].xaxis.set_visible(False)
    #                 ax[j,k].yaxis.set_visible(False)
    #                 ax[j,k].scatter(arr[:,j],arr[:,k],c = class_colours[i])
    # plt.show() #commented out this line as it'sonly for the report
    return [10,12]


def knn(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function

    # reduce the train set to the features we selected
    selected_features = [10,12]
    reduced_train_set = train_set[:, selected_features[0]]
    reduced_test_set = test_set[:, selected_features[0]]

    for feature in range(1, len(selected_features)):
        reduced_train_set = np.column_stack((reduced_train_set, train_set[:, selected_features[feature]]))
        reduced_test_set = np.column_stack((reduced_test_set, test_set[:, selected_features[feature]]))
    classifiedTests = [] # stores the result of classification for each data sample

    for test in reduced_test_set:
        closestNeighbourIndices = [] # stores the indices of the nearest neighbours found so far
        closestNeighbourClasses = [] # stores the classes of the nearest neighbours found so far

        # find the k nearest neighbours for the test sample
        for i in range(k):
            minDist = np.Infinity
            nearestClass = 0
            nearestPointIndex = 0
            for train in range(len(reduced_train_set)):
                if distance(test, reduced_train_set[train]) < minDist and train not in closestNeighbourIndices:
                    minDist = distance(test, reduced_train_set[train])
                    nearestClass = train_labels[train]
                    nearestPointIndex = train
            closestNeighbourClasses.append(nearestClass)
            closestNeighbourIndices.append(nearestPointIndex)

        # find the most common class among the neighbours (found this online)
        majorityClass = max(set(closestNeighbourClasses), key = closestNeighbourClasses.count)
        classifiedTests.append(int(majorityClass))

    #Calculate confusion matrix here
    predictions = classifiedTests
    confusion_mat = calculate_confusion_matrix(test_labels,predictions,[1,2,3])
    # confusion matrix is for the report
    # print(confusion_mat)
    # plot_matrix(confusion_mat)
    return classifiedTests

def compute_likelihood(D,mu,var):
    return stats.norm(mu,np.sqrt(var)).pdf(D)

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    selected_features = [10,12]
    reduced_train_set = train_set[:, selected_features[0]]
    reduced_test_set = test_set[:, selected_features[0]]
    for feature in range(1, len(selected_features)):
        reduced_train_set = np.column_stack((reduced_train_set, train_set[:, selected_features[feature]]))
        reduced_test_set = np.column_stack((reduced_test_set, test_set[:, selected_features[feature]]))
    #uses naive bayes classifier as alternative
    classes = [1,2,3]
    classes_featMean = []
    classes_featVar = []
    #probability of all the classes
    classes_prob = []
    #get the mean and variance of all training points belonging to a specific class
    for i in range(len(classes)):
        arr = getColours(i,reduced_train_set)
        featMean = np.mean(arr,axis = 0)
        featVar  = np.var(arr,axis = 0)
        probClass = len(arr)/len(train_labels)
        classes_prob.append(probClass)
        classes_featMean.append(featMean)
        classes_featVar.append(featVar)
    featMean = np.mean(reduced_train_set,axis = 0)
    featVar = np.var(reduced_train_set,axis = 0)

    predictions = []
    for data in reduced_test_set:
        possibilities = []
        for c in range(len(classes)):
            probClassGivenData = 1
            for i in range(len(selected_features)):
                probDataGivenClass = compute_likelihood(data[i],classes_featMean[c][i],classes_featVar[c][i])
                """commented out probData as it didn't affect the values
                    based off this: https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
                """
                # probData = compute_likelihood(data[i],featMean[i],featVar[i])
                probClassGivenData *= (probDataGivenClass)
            probClass = classes_prob[c]
            probClassGivenData *= probClass
            possibilities.append(probClassGivenData)
        pred = np.argmax(possibilities) + 1
        predictions.append(pred)
    confusion_mat = calculate_confusion_matrix(test_labels,predictions,[1,2,3])
    plot_matrix(confusion_mat)
    print(calculate_accuracy(test_labels, predictions))
    return predictions


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    selected_features = [10,12]
    reduced_train_set = train_set[:, selected_features[0]]
    reduced_test_set = test_set[:, selected_features[0]]
    for feature in range(1, len(selected_features)):
        reduced_train_set = np.column_stack((reduced_train_set, train_set[:, selected_features[feature]]))
        reduced_test_set = np.column_stack((reduced_test_set, test_set[:, selected_features[feature]]))
    cov_matrix = np.cov(reduced_train_set[:,0],reduced_train_set[:,1],rowvar = True)
    # print(cov_matrix)
    # print(reduced_train_set[:,0].shape)
    #find feature with weakly correlates with feature 10 and 12
    distances = []
    for i in range(13):
        data_set = train_set[:,i]
        cov_matrix = np.cov([reduced_train_set[:,0],reduced_train_set[:,1],data_set],rowvar = True)
        # print(cov_matrix)
        # print()
        corFeat10 = cov_matrix[0,2]
        corFeat12 = cov_matrix[1,2]
        correlation = np.array([corFeat10,corFeat12])
        # using length of a vector as a way to judge how strongly they correlate
        lengthVect = np.linalg.norm(correlation)
        distances.append(lengthVect)
        # print(correlation)
        # print()
    #selected feature should have the weakest correlation
    feature = np.argmin(distances)
    #printing 3d Cube plot

    # print(feature)
    reduced_train_set = np.column_stack((reduced_train_set, train_set[:, feature]))
    reduced_test_set = np.column_stack((reduced_test_set, test_set[:, feature]))
    # fig,ax = plt.subplots()
    # ax = fig.gca(projection = '3d')
    # ax.set_xlabel(xlabel = "feature 10")
    # ax.set_ylabel(ylabel = "feature 12")
    # ax.set_zlabel(zlabel = "feature " + str(feature))
    # plt.scatter(reduced_train_set[:,0],reduced_train_set[:,1],reduced_train_set[:,2])
    # plt.show()
    #knn algorithm
    classifiedTests = [] # stores the result of classification for each data sample

    for test in reduced_test_set:
        closestNeighbourIndices = [] # stores the indices of the nearest neighbours found so far
        closestNeighbourClasses = [] # stores the classes of the nearest neighbours found so far

        # find the k nearest neighbours for the test sample
        for i in range(k):
            minDist = np.Infinity
            nearestClass = 0
            nearestPointIndex = 0
            for train in range(len(reduced_train_set)):
                if distance(test, reduced_train_set[train]) < minDist and train not in closestNeighbourIndices:
                    minDist = distance(test, reduced_train_set[train])
                    nearestClass = train_labels[train]
                    nearestPointIndex = train
            closestNeighbourClasses.append(nearestClass)
            closestNeighbourIndices.append(nearestPointIndex)

        # find the most common class among the neighbours (found this online)
        majorityClass = max(set(closestNeighbourClasses), key = closestNeighbourClasses.count)
        classifiedTests.append(int(majorityClass))
    return classifiedTests


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    pca = PCA(n_components)
    pca = pca.fit(train_set)
    sci_data = pca.transform(train_set)
    sci_data[:,1] *= -1
    #display scatter graph
    fig,ax = plt.subplots()
    plt.title("Reduced with Scipy's PCA")
    for i in range(len(class_colours)):
        arr = getColours(i,sci_data)
        ax.scatter(arr[:,0],arr[:,1],c = class_colours[i])
    plt.show()
    #knn algorithm
    classifiedTests = [] # stores the result of classification for each data sample
    reduced_test_set = pca.transform(test_set)
    for test in reduced_test_set:
        closestNeighbourIndices = [] # stores the indices of the nearest neighbours found so far
        closestNeighbourClasses = [] # stores the classes of the nearest neighbours found so far

        # find the k nearest neighbours for the test sample
        for i in range(k):
            minDist = np.Infinity
            nearestClass = 0
            nearestPointIndex = 0
            for train in range(len(sci_data)):
                if distance(test, sci_data[train]) < minDist and train not in closestNeighbourIndices:
                    minDist = distance(test, sci_data[train])
                    nearestClass = train_labels[train]
                    nearestPointIndex = train
            closestNeighbourClasses.append(nearestClass)
            closestNeighbourIndices.append(nearestPointIndex)

        # find the most common class among the neighbours (found this online)
        majorityClass = max(set(closestNeighbourClasses), key = closestNeighbourClasses.count)
        classifiedTests.append(int(majorityClass))
    print(calculate_accuracy(test_labels,classifiedTests))
    return classifiedTests


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
        print(calculate_accuracy(test_labels, predictions))
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
        print(calculate_accuracy(test_labels, predictions))
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
