#!/usr/bin/env python3
import numpy as np

from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns=['Weighted_Price'], inplace=True)
df = df.rename(columns={'Timestamp': 'Date'})
df = df.astype({ 'Date': 'datetime64[s]' })
df = df.set_index('Date')
df['Close'] = df['Close'].ffill()

for column in ['High', 'Low', 'Open']:
    df[column] = df[column].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

df = df[df.index >= np.datetime64('2017')]

df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum',
})

df.plot()