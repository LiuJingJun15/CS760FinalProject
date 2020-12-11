# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 23:41:16 2020

@author: Jingjun Liu
"""
import pandas as pd
import numpy as np
import csv

COUNTIES = ['Brown','Dane','Eau Claire','Milwaukee','Rock','Waukesha','Winnebago']
MONTHS = ['June','July','Aug','Sep','Oct','Nov']
WEATHER = ['Temperature','Dew Point','Humidity', 'Wind Speed','Pressure','Precipitation']
POPULATION = {'Brown':264542,'Dane':546695,'Eau Claire':104646,'Milwaukee':945726,'Rock':163354,'Waukesha':404198,'Winnebago':171907}
AREA = {'Brown':529.71,'Dane':1197.24,'Eau Claire':637.98,'Milwaukee':241.40,'Rock':718.14,'Waukesha':549.57 ,'Winnebago':434.49 }
POPULATION_DENTISY = {'Brown':410.4,'Dane':406.2,'Eau Claire':155.9,'Milwaukee':801.5,'Rock':221.0,'Waukesha':676.0,'Winnebago':291.1}

# cum_cases_wisc_df = pd.read_csv("../data/Cumulative_cases_data_wisc.csv") 
# cum_cases_wisc = cum_cases_wisc_df[:].to_numpy()

# weather_df = pd.read_excel('../data/counties/Brown/June/weather.xlsx')
# weather = weather_df[:].to_numpy()

# main part to make data
# date, county, new case ratio, all weather indicators
def make_data():
    cum_cases_wisc_df = pd.read_csv("../data/Cumulative_cases_data_wisc.csv") 
    cum_cases_wisc = cum_cases_wisc_df[:].to_numpy()
    new_cases_wisc = choose_covid_data(cum_cases_wisc, '2020/6/1/')
    shape = np.shape(new_cases_wisc)
    days = shape[0]
    new_cases_county_list = []
    for county in COUNTIES:
        covid_data_path = '../data/counties/'+county+'/'+'Cumulative_cases_data_'+county +'.csv'
        cum_cases_county = pd.read_csv(covid_data_path) 
        cum_cases_county = cum_cases_county[:].to_numpy()
        new_cases_county = choose_covid_data(cum_cases_county, '2020/6/1/')
        new_cases_county_ratio = new_cases_county.copy()
        for i in range(days):
            new_cases_county_ratio[i,2] = new_cases_county[i,2]/new_cases_wisc[i,2]/POPULATION_DENTISY[county]
        weather_array = choose_weather_all(county)
        weather_shape = np.shape(weather_array)
        # print(county, weather_shape[0])
        # print(shape[0])
        if weather_shape[0] > shape[0]:   
            d = weather_shape[0] - shape[0]
            # print(weather_shape[0]-d,weather_shape[0])
            weather_array = np.delete(weather_array, range(weather_shape[0]-d,weather_shape[0]), 0)
        elif weather_shape[0] < shape[0]:
            d = shape[0] - weather_shape[0]
            # print(weather_shape[0]-d,weather_shape[0])
            new_cases_county_ratio = np.delete(new_cases_county_ratio, range(shape[0]-d,shape[0]), 0)
        # print(np.shape(weather_array))
        new_cases_county_ele = np.concatenate((new_cases_county_ratio, weather_array), axis=1)
        new_cases_county_list.append(new_cases_county_ele)
    result = np.empty(shape = (0,9))
    for i in range(7):
        result = np.concatenate((result, new_cases_county_list[i]), axis=0)
    resultShape = np.shape(result)
    finalResult = result.copy()
    for i in range(3, resultShape[1]):
        for j in range(13, resultShape[0]):
            finalResult[j,i] = 1/7*(result[j-13,i] +result[j-12,i]+result[j-11,i]+result[j-10,i]+result[j-9,i] +result[j-8,i]+result[j-7,i])
    for i in range(resultShape[0]):
        finalResult[i,2] = float(finalResult[i,2])
    return finalResult
    
def make_data_of_county(county):
    cum_cases_wisc_df = pd.read_csv("../data/Cumulative_cases_data_wisc.csv") 
    cum_cases_wisc = cum_cases_wisc_df[:].to_numpy()
    new_cases_wisc = choose_covid_data(cum_cases_wisc, '2020/6/1/')
    shape = np.shape(new_cases_wisc)
    days = shape[0]
    covid_data_path = '../data/counties/'+county+'/'+'Cumulative_cases_data_'+county +'.csv'
    cum_cases_county = pd.read_csv(covid_data_path) 
    cum_cases_county = cum_cases_county[:].to_numpy()
    new_cases_county = choose_covid_data(cum_cases_county, '2020/6/1/')
    new_cases_county_ratio = new_cases_county.copy()
    for i in range(days):
        new_cases_county_ratio[i,2] = new_cases_county[i,2]/new_cases_wisc[i,2]/POPULATION_DENTISY[county]
    weather_array = choose_weather_all(county)
    weather_shape = np.shape(weather_array)
    # print(county, weather_shape[0])
    # print(shape[0])
    if weather_shape[0] > shape[0]:   
        d = weather_shape[0] - shape[0]
        # print(weather_shape[0]-d,weather_shape[0])
        weather_array = np.delete(weather_array, range(weather_shape[0]-d,weather_shape[0]), 0)
    elif weather_shape[0] < shape[0]:
        d = shape[0] - weather_shape[0]
        # print(weather_shape[0]-d,weather_shape[0])
        new_cases_county_ratio = np.delete(new_cases_county_ratio, range(shape[0]-d,shape[0]), 0)
    # print(np.shape(weather_array))
    result = np.concatenate((new_cases_county_ratio, weather_array), axis=1)
    resultShape = np.shape(result)
    finalResult = result.copy()
    for i in range(3, resultShape[1]):
        for j in range(13, resultShape[0]):
            finalResult[j,i] = 1/7*(result[j-13,i] +result[j-12,i]+result[j-11,i]+result[j-10,i]+result[j-9,i] +result[j-8,i]+result[j-7,i])
    for i in range(resultShape[0]):
        finalResult[i,2] = float(finalResult[i,2])
    return finalResult

# date, county, new cases
def choose_covid_data(data_array, dateStr):
    startIndex = np.where(data_array[:,0] == dateStr)
    startIndex = startIndex[0]
    arrShape = np.shape(data_array)
    numLine = arrShape[0]-startIndex+1
    numLine = numLine[0]
    resultArr = np.empty(shape=(numLine, 3), dtype=object)
    for i in range(numLine - 1):
        resultArr[i,0] = data_array[i+startIndex,0]
        resultArr[i,1] = data_array[i+startIndex,2]
        resultArr[i,2] = data_array[i+startIndex,3] - data_array[i+startIndex-1,3]
    resultArr = np.delete(resultArr, -1,0)
    return resultArr


# 'Temperature','Dew Point','Humidity', 'Wind Speed','Pressure','Precipitation'
def choose_weather(weather_array):
    arrShape = np.shape(weather_array)
    numLine = arrShape[0] - 1
    resultArr = np.empty(shape=(numLine, 6), dtype=object)
    resultArr[:,0] = weather_array[1:,2]
    resultArr[:,1] = weather_array[1:,5]
    resultArr[:,2] = weather_array[1:,8]
    resultArr[:,3] = weather_array[1:,11]
    resultArr[:,4] = weather_array[1:,14]
    resultArr[:,5] = weather_array[1:,16]
    return resultArr


def choose_weather_all(county):
    weather_list = []
    weather_path1 = '../data/counties/' + county + '/'
    for month in MONTHS:
        # print('aaaaaa')
        # print(type(county), type(month))
        weather_path =  weather_path1 + month + '/weather.xlsx'
        # print(type(weather_path))
        weather_df = pd.read_excel(weather_path)
        weather_array = weather_df[:].to_numpy()
        weather = choose_weather(weather_array)
        weather_list.append(weather)
    result = np.empty(shape = (0,6))
    for i in range(6):
        result = np.concatenate((result, weather_list[i]), axis=0)
    return result


def write_data_csv(county):
    filename = "./traindata_regression_"+county+".csv"
    data = make_data_of_county(county)
    traindata_regression = data[:,2:]
    pd.DataFrame(traindata_regression).to_csv(filename,header=None, index=None)

for county in COUNTIES:
    write_data_csv(county)
        