# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 19:19:55 2020

@author: LIUJINGJUN
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import neighbors
import numpy as np
import random


def linear_regression_model(data_regression):
    # data_regression = data_regression[:].to_numpy()
    X = data_regression[:,1:]
    Y = data_regression[:,0]
    nSample = np.shape(Y)
    nSample = nSample[0]
    bias = np.ones(shape = (nSample,1))
    X = np.concatenate((X,bias), axis = 1)
    X_T = np.transpose(X)
    if np.linalg.matrix_rank(np.dot(X_T,X)) == 7:
        # print("Here!!!!!!!!!!!!!!!!!!")
        theta = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)),X_T),Y)
    else:
        X = np.delete(X,5,1)
        X_T = np.transpose(X)
        theta = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)),X_T),Y)
        theta = np.insert(theta, 5, 0);
    return theta

def weighted_res(response, distance):
    n = len(response)
    sumres = 0
    dis_copy = distance.copy()
    for i in range(n):
        if distance[i] == 0:
            return response[i]
        else:
            dis_copy[i] = 1/distance[i]
    for i in range(n):
        sumres += response[i]*dis_copy[i]
    result = sumres/sum(dis_copy)
    # print(result)
    return result

def weighted_dis(a,b):
    return np.linalg.norm(a-b)

def weighted_KNN_predictor(testData, new_x):
    final_response = [1e-5]*5
    final_distance = [1e5]*5
    Y = testData[:,0]
    X = testData[:,1:]
    nSample = np.shape(Y)
    nSample = nSample[0]
    for i in range(nSample):
        currX = X[i,:]
        currDist = weighted_dis(currX, new_x)
        currRes = Y[i]
        for j in range(5):
            if final_distance[j] > currDist:
                if j == 4:
                    final_distance[j] = currDist
                    final_response[j] = currRes
                else:
                    for k in range(4,j,-1):
                        final_distance[k] = final_distance[k-1]
                        final_response[k] = final_response[k-1]
                    final_distance[j] = currDist
                    final_response[j] = currRes
                break
            else:
                continue
    return weighted_res(final_response,final_distance)


