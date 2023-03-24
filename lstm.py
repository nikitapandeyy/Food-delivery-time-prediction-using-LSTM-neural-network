# Importing required libraries
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Loading the dataset
df = pd.read_csv('food_delivery_data.csv')

# Preprocessing the dataset
df = df.drop(['Order ID', 'Restaurant', 'Rider'], axis=1)
df['Time from Pickup to Arrival'] = df['Time from Pickup to Arrival'] / 60
X = df.drop(['Time from Pickup to Arrival'], axis=1).values
y = df['Time from Pickup to Arrival'].values.reshape(-1, 1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating the LSTM neural network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=100, batch_size=32,
                    validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
                    callbacks=[early_stop])

# Visualizing the training and validation loss
fig = px.line(history.history, y=['loss', 'val_loss'], labels={'value': 'Loss', 'variable': 'Type', 'index': 'Epoch'},
              title='Training and Validation Loss')
fig.show()

# Evaluating the model
y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
rmse = np.sqrt(np.mean(np.power(scaler.inverse_transform(y_test) - scaler.inverse_transform(y_pred), 2)))
print('Root Mean Squared Error:', rmse)

# Saving the model
model.save('food_delivery_time_prediction_model.h5')
