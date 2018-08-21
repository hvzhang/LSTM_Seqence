# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:33:37 2018

@author: zhanghui183140
"""


import pandas as pd
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#%matplotlib inline


#Read the raw data
Train_Data_File = 'D:\\workspace\\py\\yanfang\\hd\\TrainData.csv'
Test_Data_File = 'D:\\workspace\\py\\yanfang\\hd\\TestData.csv'

#Load & prepare the training data
TrainData = pd.read_csv(Train_Data_File, sep=",", header=None)
TrainData.drop(TrainData.columns[[5, 6]], axis=1, inplace=True)
TrainData.drop(TrainData.columns[[0,1,2,3]], axis=1, inplace=True)

trainDataSet = TrainData.values

#Load and prepare the test data
TestData = pd.read_csv(Test_Data_File, sep=",", header=None)
TestData.drop(TestData.columns[[5, 6]], axis=1, inplace=True)
TestData.drop(TestData.columns[[0,1,2,3]], axis=1, inplace=True)

testDataSet = TestData.values



#print(TrainData)

#plt.plot( testDataSet)
#plt.show()


# X is the number of passengers at a given time (t) and Y is the number of passengers at the next time (t + 1).

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)


# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(trainDataSet, look_back)
testX, testY = create_dataset(testDataSet, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#print(trainX)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

'''
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
'''
# plot baseline and predictions
#plt.plot(scaler.inverse_transform(dataset))
#plt.plot(trainPredictPlot)
#plt.plot(testPredictPlot)
#plt.show()

