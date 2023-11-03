import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

print("Number of logical cores: ", os.cpu_count())
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# load data
ticker = "META"

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2019, 1, 1)

data = yf.Ticker(ticker).history(start=start, end=end)

# prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(data["Close"].values.reshape(-1,1))

prediction_days = 120

x_train = []
y_train = []

for x in range(prediction_days, len(scaledData)):
    x_train.append(scaledData[x - prediction_days:x, 0])
    y_train.append(scaledData[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # rows, columns and add a z axis also

#build the model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=35, batch_size=128)

# load test data

test_start = dt.datetime(2019, 1, 2)
test_end = dt.datetime(2023, 10, 20)

test_data = yf.Ticker(ticker).history(start=test_start, end=test_end)
actual_prices = test_data["Close"].values
total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = predicted_prices.flatten()

adjusted_predicted_prices = [predicted_prices[0]]

errors = actual_prices - predicted_prices
errors = np.array(errors)
X = errors[:-1].reshape(-1, 1)
Y = errors[1:]

regressor = LinearRegression()
regressor.fit(X, Y)
alpha = regressor.coef_[0]

mean_error = 0
error_acc = 0

for i in range(0, len(predicted_prices)):
    error = actual_prices[i] - predicted_prices[i]
    error_acc += error

mean_error = error_acc/len(predicted_prices)

adjusted_predicted_prices = np.empty(len(predicted_prices))

for i in range(0, len(predicted_prices)):
    adjusted_predicted_prices[i] = predicted_prices[i] + alpha*mean_error


plt.plot(actual_prices, color="black", label=f"actual {ticker} price")
plt.plot(predicted_prices, color="green", label=f"predicted {ticker} price")
plt.plot(adjusted_predicted_prices, color="blue", label=f"adjusted {ticker} price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()










