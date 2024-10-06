import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
# Step 1: Download the dataset
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2023-01-01')
data = data[['Close']]
# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
# Step 3: Create sequences
sequence_length = 60
X = []
y = []
for i in range(sequence_length, len(scaled_data)):
 X.append(scaled_data[i-sequence_length:i, 0])
 y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
# Step 4: Split the data
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
# Step 5: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))
# Step 6: Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)
# Step 7: Evaluate the model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
# Plotting the results
plt.figure(figsize=(16, 8))
plt.plot(data.index[train_size + sequence_length:], data['Close'][train_size + sequence_length:],
color='blue', label='Actual Prices')
plt.plot(data.index[train_size + sequence_length:], predictions, color='red', label='Predicted
Prices')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# Step 8: Make future predictions (optional)
# Use the last sequence to predict the next price
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.reshape(last_sequence, (1, sequence_length, 1))
next_price_scaled = model.predict(last_sequence)
next_price = scaler.inverse_transform(next_price_scaled)
print(f'The predicted next price for {ticker} is: {next_price[0, 0]:.2f} USD')
