import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from functools import reduce

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

# build the first LSTM model
model1 = Sequential()

model1.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50, return_sequences=False))
model1.add(Dropout(0.2))
model1.add(Dense(units=1))

model1.compile(optimizer="adam", loss="mean_squared_error")
model1.fit(x_train, y_train, epochs=35, batch_size=128)

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

predicted_prices = model1.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = predicted_prices.flatten()

# Calculate errors
errors = actual_prices - predicted_prices

# Prepare the data for the second LSTM model
x_error_train = []
y_error_train = []

for x in range(prediction_days, len(errors)):
    x_error_train.append(errors[x - prediction_days:x])
    y_error_train.append(errors[x])

x_error_train, y_error_train = np.array(x_error_train), np.array(y_error_train)
x_error_train = np.reshape(x_error_train, (x_error_train.shape[0], x_error_train.shape[1], 1))

# Build the second LSTM model for error prediction
model2 = Sequential()

model2.add(LSTM(units=50, return_sequences=True, input_shape=(x_error_train.shape[1], 1)))
model2.add(Dropout(0.2))
model2.add(LSTM(units=50, return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(units=50, return_sequences=False))
model2.add(Dropout(0.2))
model2.add(Dense(units=1))

model2.compile(optimizer="adam", loss="mean_squared_error")
model2.fit(x_error_train, y_error_train, epochs=35, batch_size=128)

# Make predictions with the second LSTM model
predicted_errors = model2.predict(x_error_train)
predicted_errors = predicted_errors.flatten()

# Adjust the predicted stock prices
adjusted_predicted_prices = np.empty_like(predicted_prices)
adjusted_predicted_prices[:prediction_days - 1] = predicted_prices[:prediction_days - 1]
adjusted_length = min(len(predicted_prices[prediction_days - 1:]), len(predicted_errors))
adjusted_predicted_prices[prediction_days - 1:prediction_days - 1 + adjusted_length] = predicted_prices[prediction_days - 1:prediction_days - 1 + adjusted_length] + predicted_errors[:adjusted_length]

rse1 = []
rse2 = []
rmse1 = 0
rmse2 = 0

predicted_prices = predicted_prices.flatten()

for i in range(0, len(predicted_prices)):
    rse1.append(((actual_prices[i] - predicted_prices[i])**2)**1/2)
    rse2.append(((actual_prices[i] - adjusted_predicted_prices[i])**2)**1/2)

adjusted_predicted_prices = adjusted_predicted_prices[:-2]
rse2 = rse2[:-2]

rmse1 = reduce(lambda x, y: x + y, predicted_prices)/len(predicted_prices)
rmse2 = reduce(lambda x, y: x + y, adjusted_predicted_prices)/len(adjusted_predicted_prices)

print(f"predicted prices RMS error - {rmse1}")
print(f"adjusted prices RMS error - {rmse2}")

# Plot the results
fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(actual_prices, color="black", label="actual price")
axes[0, 0].plot(predicted_prices, color="green", label="predicted price")
axes[0, 0].set_xlabel("Days")
axes[0, 0].set_ylabel("Price")
axes[0, 0].legend()

axes[0, 1].plot(actual_prices, color="black", label="actual price")
axes[0, 1].plot(adjusted_predicted_prices, color="blue", label="adjusted price")
axes[0, 1].set_xlabel("Days")
axes[0, 1].set_ylabel("Price")
axes[0, 1].legend()

axes[1, 0].plot(rse1, color="green", label="predicted error")
axes[1, 0].plot(rse2, color="blue", label="adjusted error")
axes[1, 0].set_xlabel("Days")
axes[1, 0].set_ylabel("root square error")
axes[1, 0].legend()

fig.delaxes(axes[1, 1])
plt.tight_layout()
plt.show()
