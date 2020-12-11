# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:24:14 2020

@author: LIUJINGJUN
"""

import makeData
import learn 
import matplotlib.pyplot as plt
import csv
import pandas as pd
import random
import numpy as np

COUNTIES = ['Brown','Dane','Eau Claire','Milwaukee','Rock','Waukesha','Winnebago']
WEATHER = ['Temperature','Dew Point','Humidity', 'Wind Speed','Pressure','Precipitation']


def linear_reg_plot_state():
    filename = "./traindata_regression.csv"
    data_regression = pd.read_csv(filename)
    data_regression = data_regression[:].to_numpy()
    data_copy = data_regression.copy()
    dataShape = np.shape(data_copy)
    Y = data_regression[:,0]
    nSample = np.shape(Y)
    nSample = nSample[0]
    ntest = int(np.floor(nSample/5))
    ntrain = nSample - ntest 
    testIdx = np.sort(random.sample(range(nSample), ntest))
    testData = data_copy[testIdx,:]
    trainData = data_regression.copy()
    trainData = np.delete(trainData,testIdx,0)
    bias_train = np.ones(shape = (ntrain,1))
    bias_test = np.ones(shape = (ntest,1))
    testData = np.concatenate((testData,bias_test), axis = 1)
    theta = linear_regression_model(trainData)
    y_pred = np.dot(theta, np.transpose(testData[:,1:]))
    plt.plot(testData[:,0],'r*',label = 'real response')
    plt.plot(y_pred,'bs', label = 'predicted response')
    plt.legend()
    
def linear_reg_cv_state():
    nCorrect = 0
    divSum = 0
    filename = "./traindata_regression.csv"
    data_regression = pd.read_csv(filename)
    data_regression = data_regression[:].to_numpy()
    data_copy = data_regression.copy()
    dataShape = np.shape(data_copy)
    Y = data_regression[:,0]
    nSample = np.shape(Y)
    nSample = nSample[0]
    ntest = int(np.floor(nSample/5))
    ntrain = nSample - ntest 
    testIdx = np.sort(random.sample(range(nSample), ntest))
    testData = data_copy[testIdx,:]
    trainData = data_regression.copy()
    trainData = np.delete(trainData,testIdx,0)
    bias_train = np.ones(shape = (ntrain,1))
    bias_test = np.ones(shape = (ntest,1))
    testData = np.concatenate((testData,bias_test), axis = 1)
    theta = linear_regression_model(trainData)
    y_pred = np.dot(theta, np.transpose(testData[:,1:]))
    for j in range(ntest):
        if testData[j,0] == 0:
            div = 0.5
        else:
            div = abs((y_pred[j]-testData[j,0])/testData[j,0])
        # print(div)
        if div<0.3:
            nCorrect += 1
        divSum += div
    return nCorrect/ntest, divSum/ntest
    
def linear_reg_plot_county():
    i = 0
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Linear Regression Prediction for county")
    plt.axis('off')
    accuracy = {}
    deviation = {}
    for county in COUNTIES:
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        data_copy = data_regression.copy()
        dataShape = np.shape(data_copy)
        nSample = dataShape[0]
        ntest = int(np.floor(nSample/5))
        ntrain = nSample - ntest 
        testIdx = np.sort(random.sample(range(nSample), ntest))
        testData = data_copy[testIdx,:]
        trainData = data_regression.copy()
        trainData = np.delete(trainData,testIdx,0)
        bias_test = np.ones(shape = (ntest,1))
        testData = np.concatenate((testData,bias_test), axis = 1)
        theta = linear_regression_model(trainData)
        Y_pred = np.dot(theta, np.transpose(testData[:,1:]))
        if i == 0:
            axs[int(i/2), i%2].plot(testData[:,0],'r*',label = 'real response')
            axs[int(i/2), i%2].plot( Y_pred,'b.', label = 'predicted response')
        else:
            axs[int(i/2), i%2].plot(testData[:,0],'r*')
            axs[int(i/2), i%2].plot( Y_pred,'b.')
        axs[int(i/2), i%2].set_title(county)
        i+=1
    for ax in fig.get_axes():
        ax.label_outer()
    fig.legend(loc='lower right')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)


def linear_reg_cv_county():
    i = 0
    accuracy = {}
    deviation = {}
    for county in COUNTIES:
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        data_copy = data_regression.copy()
        dataShape = np.shape(data_copy)
        nSample = dataShape[0]
        ntest = int(np.floor(nSample/5))
        ntrain = nSample - ntest 
        testIdx = np.sort(random.sample(range(nSample), ntest))
        testData = data_copy[testIdx,:]
        trainData = data_regression.copy()
        trainData = np.delete(trainData,testIdx,0)
        bias_test = np.ones(shape = (ntest,1))
        testData = np.concatenate((testData,bias_test), axis = 1)
        theta = linear_regression_model(trainData)
        Y_pred = np.dot(theta, np.transpose(testData[:,1:]))
        nCorrect = 0
        divSum = 0
        for j in range(ntest):
            if testData[j,0] == 0:
                div = 0.5
            else:
                div = abs((Y_pred[j]-testData[j,0])/testData[j,0])
            # print(div)
            if div<0.3:
                nCorrect += 1
            divSum += div
        accuracy[county] = nCorrect/ntest
        deviation[county] = divSum/ntest
    return accuracy, deviation


def train_and_get_param_county(model):
    i = 0
    params = {}
    # params.keys = COUNTIES
    for county in COUNTIES:
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        trained_model = model(data_regression)
        params[county] = trained_model.get_params()
    return params

def significant_features_linear_reg_county():
    theta = {}
    mu = []
    finalVal = {}
    # params.keys = COUNTIES
    i = 0
    for county in COUNTIES:
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        X = data_regression[:,1:]
        Y = data_regression[:,0]
        # Y = np.reshape(Y, (177,1))
        nSample = np.shape(Y)
        nSample = nSample[0]
        bias = np.ones(shape = (nSample,1))
        X = np.concatenate((X,bias), axis = 1)
        X_T = np.transpose(X)
        # print(np.shape(np.dot(X_T,X)))
        if np.linalg.matrix_rank(np.dot(X_T,X)) == 7:
            theta_this = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)),X_T),Y)
            theta[county] = theta_this
            Y = np.reshape(Y, (177,1))
            sub = Y - np.dot(X,theta_this)
            sd = 1/nSample*(np.dot(np.transpose(sub),sub))
            sd_Matrix = np.linalg.inv(np.dot(X_T,X))
            result = []
            for j in range(7):
                # mu.append(sd_Matrix[i,i])
                result.append(theta_this[j]/sd_Matrix[j,j])
            finalVal[county] = result
            i += 1
        else:
            X = np.delete(X,5,1)
            # print(np.shape(X))
            X_T = np.transpose(X)
            # print(np.shape(np.dot(X_T,X)))
            theta_this = np.dot(np.dot(np.linalg.inv(np.dot(X_T,X)),X_T),Y)
            theta[county] = theta_this
            Y = np.reshape(Y, (nSample,1))
            sub = Y - np.dot(X,theta_this)
            sd = 1/nSample*(np.dot(np.transpose(sub),sub))
            sd_Matrix = np.linalg.inv(np.dot(X_T,X))
            result = []
            for j in range(6):
                result.append(theta_this[j]/sd_Matrix[j,j])
            finalVal[county] = result
            i += 1
    return theta, finalVal

def linear_reg_validation_county():
    theta, comp = significant_features_linear_reg()
    for county in COUNTIES:
        if np.shape(theta[county])[0] < 7:
            theta[county] = np.insert(theta[county],5,0)
        theta[county] = np.reshape(theta[county] ,(1,7))
    accu = {}
    dev = {}
    for county in COUNTIES:
        accu[county] = 0
        dev[county] = 0
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        Y = data_regression[:,0]
        nSample = np.shape(Y)
        nSample = nSample[0]
        bias = np.ones(shape = (nSample,1))
        data_regression = np.concatenate((data_regression,bias), axis = 1)
        Y_pred = np.dot(theta[county], np.transpose(data_regression[:,1:]))
        # print(np.shape(Y_pred))
        # print(np.shape(Y))
        nCorrect = 0
        divSum = 0
        # print(Y_pred[0,20])
        for j in range(nSample):
            if data_regression[j,0] == 0:
                div = 0.5
            else:
                # print((Y_pred[1,j]-Y[j]))
                div = abs((Y_pred[0,j]-Y[j])/Y[j])
                # print(div)
            if div < 0.3:
                nCorrect += 1
            divSum += div
        accu[county] = nCorrect/nSample
        dev[county] = divSum/nSample
    return accu, dev

################################ KNN **********************************
def KNN_plot_state():
    filename = "./traindata_regression.csv"
    data_regression = pd.read_csv(filename)
    data_regression = data_regression[:].to_numpy()
    data_copy = data_regression.copy()
    dataShape = np.shape(data_copy)
    Y = data_regression[:,0]
    nSample = np.shape(Y)
    nSample = nSample[0]
    ntest = int(np.floor(nSample/5))
    ntrain = nSample - ntest 
    testIdx = np.sort(random.sample(range(nSample), ntest))
    testData = data_copy[testIdx,:]
    trainData = data_regression.copy()
    trainData = np.delete(trainData,testIdx,0)
    y_pred = np.zeros((1,ntest))
    for i in range(ntest):
        x_new = testData[i,1:]
        y_real = testData[i,0]
        # print(y_real)
        y_pred[0,i] = weighted_KNN_predictor(trainData, x_new)
    plt.plot(testData[:,0],'r*',label = 'real response')
    plt.plot(y_pred[0,:],'bs', label = 'predicted response')
    plt.legend()
    plt.title("KNN prediction result for the whole state")
    
def KNN_cv_state():
    nCorrect = 0
    devSum = 0
    filename = "./traindata_regression.csv"
    data_regression = pd.read_csv(filename)
    data_regression = data_regression[:].to_numpy()
    data_copy = data_regression.copy()
    dataShape = np.shape(data_copy)
    Y = data_regression[:,0]
    nSample = np.shape(Y)
    nSample = nSample[0]
    ntest = int(np.floor(nSample/5))
    ntrain = nSample - ntest 
    testIdx = np.sort(random.sample(range(nSample), ntest))
    testData = data_copy[testIdx,:]
    trainData = data_regression.copy()
    trainData = np.delete(trainData,testIdx,0)
    y_pred = np.zeros((1,ntest))
    for i in range(ntest):
        x_new = testData[i,1:]
        y_real = testData[i,0]
        # print(y_real)
        y_pred = weighted_KNN_predictor(trainData, x_new)
        if y_real == 0:
                div = 0.5
        else:
            div=abs((y_pred-y_real)/y_real)
        if div<0.3:
            nCorrect += 1
        devSum += div
    accuracy = nCorrect/ntest
    deviation = devSum/ntest
    return accuracy, deviation

def KNN_cv_state_n(n):
    sumAccu = 0
    sumDev = 0
    for i in range(n):
        accu, dev = KNN_cv_state()
        sumAccu += accu
        sumDev+= dev
    return sumAccu/n, sumDev/n

def KNN_plot_county():
    i = 0
    fig, axs = plt.subplots(4, 2)
    fig.suptitle("Distance-weighted KNN prediction result for county")
    plt.axis('off')
    accuracy = {}
    deviation = {}
    for county in COUNTIES:
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        data_copy = data_regression.copy()
        dataShape = np.shape(data_copy)
        nSample = dataShape[0]
        ntest = int(np.floor(nSample/5))
        ntrain = nSample - ntest 
        testIdx = np.sort(random.sample(range(nSample), ntest))
        testData = data_copy[testIdx,:]
        trainData = data_regression.copy()
        trainData = np.delete(trainData,testIdx,0)
        y_pred = np.zeros((1,ntest))
        for j in range(ntest):
            x_new = testData[j,1:]
            y_real = testData[j,0]
            # print(y_real)
            y_pred[0,j] = weighted_KNN_predictor(trainData, x_new)
        if i == 0:
            axs[int(i/2), i%2].plot(testData[:,0],'r*',label = 'real response')
            axs[int(i/2), i%2].plot( y_pred[0,:],'b.', label = 'predicted response')
        else:
            axs[int(i/2), i%2].plot(testData[:,0],'r*')
            axs[int(i/2), i%2].plot( y_pred[0,:],'b.')
        axs[int(i/2), i%2].set_title(county)
        i += 1
    for ax in fig.get_axes():
        ax.label_outer()
    fig.legend(loc='lower right')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
            
    
def KNN_validate_county():
    accuracy = {}
    deviation = {}
    for county in COUNTIES:
        accuracy[county] = 0
        deviation[county] = 0
        nCorrect = 0
        devSum = 0
        filename = "./traindata_regression_"+county+".csv"
        data_regression = pd.read_csv(filename)
        data_regression = data_regression[:].to_numpy()
        data_copy = data_regression.copy()
        Y = data_regression[:,0]
        nSample = np.shape(Y)
        nSample = nSample[0]
        ntest = int(np.floor(nSample/5))
        ntrain = nSample - ntest 
        testIdx = np.sort(random.sample(range(nSample), ntest))
        testData = data_copy[testIdx,:]
        trainData = data_regression.copy()
        trainData = np.delete(trainData,testIdx,0)
        for i in range(ntest):
            x_new = testData[i,1:]
            y_real = testData[i,0]
            # print(y_real)
            y_pred = weighted_KNN_predictor(trainData, x_new)
            if y_real == 0:
                div = 0.5
            else:
                div=abs((y_pred-y_real)/y_real)
            if div<0.3:
                nCorrect += 1
            devSum += div
        accuracy[county] = nCorrect/ntest
        deviation[county] = devSum/ntest
    return accuracy, deviation
        
        
def KNN_validate_county_ntimes(n_times):
    sumAccu = {} 
    sumDev= {} 
    for county in COUNTIES:
        sumAccu[county] = 0
        sumDev[county] = 0
        for i in range(n_times):
            accu, dev = KNN_validate_county()
            sumAccu[county] += accu[county]
            sumDev[county]+= dev[county]
    for county in COUNTIES:
        sumAccu[county] /= n_times
        sumDev[county] /= n_times
    return sumAccu, sumDev

