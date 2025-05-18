import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os


def load_and_preprocess_data(file_path):
    """Load and preprocess BTC dataset."""
    df = pd.read_csv(file_path)

    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Select useful columns
    df = df[['timestamp', 'close']]

    # Fill missing values using forward fill
    df.fillna(method='ffill', inplace=True)

    # Normalize 'close' prices using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['close'] = scaler.fit_transform(df[['close']])

    return df, scaler


def create_sequences(data, seq_length=24*60):
    """Create sequences for training (24-hour windows)."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


def save_preprocessed_data():
    """Preprocess and save data."""
    coinbase_file = "data/coinbase.csv"
    bitstamp_file = "data/bitstamp.csv"

    # Load and preprocess both datasets
    coinbase_df, coinbase_scaler = load_and_preprocess_data(coinbase_file)
    bitstamp_df, bitstamp_scaler = load_and_preprocess_data(bitstamp_file)

    # Create sequences
    seq_length = 24 * 60  # 24 hours of minute data
    X_coin, y_coin = create_sequences(coinbase_df['close'].values, seq_length)
    X_bit, y_bit = create_sequences(bitstamp_df['close'].values, seq_length)

    # Save processed data
    np.save("data/X_coin.npy", X_coin)
    np.save("data/y_coin.npy", y_coin)
    np.save("data/X_bit.npy", X_bit)
    np.save("data/y_bit.npy", y_bit)

    print("Data preprocessing complete.")


if __name__ == "__main__":
    save_preprocessed_data()