#!/usr/bin/env python3
import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1_filter = (df1['Timestamp'] >= 1417411980) & (df1['Timestamp'] <= 1417417980)
df2_filter = (df2['Timestamp'] >= 1417411980) & (df2['Timestamp'] <= 1417417980)

df1 = df1[df1_filter]
df2 = df2[df2_filter]

df1['Exchange'] = 'coinbase'
df2['Exchange'] = 'bitstamp'

df = pd.concat([df2, df1], axis=0)
df = df.groupby(['Timestamp', 'Exchange']).sum()

print(df)