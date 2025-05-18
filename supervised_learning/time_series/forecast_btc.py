import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
X_coin = np.load("data/X_coin.npy")
y_coin = np.load("data/y_coin.npy")

# Reshape input data for LSTM (samples, time steps, features)
X_coin = X_coin.reshape((X_coin.shape[0], X_coin.shape[1], 1))

# Define RNN model


def build_rnn_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_coin.shape[1], 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Predict next closing price
    ])

    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model


# Train the model
model = build_rnn_model()
model.fit(X_coin, y_coin, epochs=10, batch_size=32, validation_split=0.1)

# Save the trained model
model.save("btc_forecast_model.h5")
print("Model training complete and saved.")