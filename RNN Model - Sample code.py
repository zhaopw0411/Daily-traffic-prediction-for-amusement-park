# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


## this one is for you

#import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
 #from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


usage = pd.read_csv("usage.csv")
weather = pd.read_csv("weather.csv")

usage_weather = pd.merge(usage,weather, left_on = "Date", right_on="Date")


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(usage_weather.iloc[:,1:6])
df = pd.DataFrame(scaled , columns = usage_weather.columns[1:])


#split into train and test sets
values_df = df.values
n_train_hours = 1369
train = values_df[:n_train_hours, :]
test = values_df[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, [1,2,3,4]], train[:, 0]
test_X, test_y = test[:, [1,2,3,4]], test[:, 0]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),go_backwards = True, activation = 'tanh'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=500, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
  
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, :]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, :]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


