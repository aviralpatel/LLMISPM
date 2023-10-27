import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from keys import newsAPIKey

# load data
ticker = "AAPL"

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2019, 1, 1)

data = yf.Ticker(ticker).history(start=start, end=end)
dict_data = data.to_dict()

for i in range(0, len(dict_data["Close"])):
    if day > 365:
        day = 1
    dayLst.append(day)
    day += 1


# prepare the data
scaledData = MinMaxScaler(feature_range=(0,1)).fit_transform(data["Close"].values.reshape(-1,1))

prediction_days = 120

x_train = []
y_train = []

for x in range(0, len(scaledData) - prediction_days):
    x_train.append(scaledData[x:x + prediction_days, 0])
    y_train.append(scaledData[x + prediction_days, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # rows, columns and add a z axis also

#build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))





