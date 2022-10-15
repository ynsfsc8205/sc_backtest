import pandas as pd
import os
import datetime


def get_data(data_name, columns=None, start_time='2016-01-01', end_time='2021-01-01', frequency=1):
    base_data_names = ['adj_close_price', 'adj_open_price', 'adj_high_price', 'adj_low_price',
                       'volume', 'value', 'openint']
    base_columns = ['IF', 'IC', 'IH']
    if columns is None:
        columns = base_columns

    if data_name not in base_data_names:
        return

    if not isinstance(columns, list):
        return

    for c in columns:
        if c not in base_columns:
            print(f'No {c} Data')
            return

    if 240 % frequency != 0:
        print(F'240%frequency should be 0 and frequency should be smaller than or equal to 240')
        return

    dirname, _ = os.path.split(os.path.abspath(__file__))
    path = os.path.join(dirname, 'dataset', f'_{data_name}.csv')
    output = pd.read_csv(path, index_col=0, header=0, converters={'date': pd.to_datetime})
    output = output.loc[pd.to_datetime(start_time):pd.to_datetime(end_time), columns]

    if frequency <= 240:
        if data_name in ['adj_close_price', 'openint']:
            output = output.iloc[frequency-1::frequency]
        elif data_name in ['adj_open_price']:
            output = output.rolling(frequency).apply(lambda x: x.iloc[0]).iloc[frequency-1::frequency]
        elif data_name in ['adj_high_price']:
            output = output.rolling(frequency).max().iloc[frequency - 1::frequency]
        elif data_name in ['adj_low_price']:
            output = output.rolling(frequency).min().iloc[frequency - 1::frequency]
        elif data_name in ['volume', 'value']:
            output = output.rolling(frequency).sum().iloc[frequency - 1::frequency]

    if frequency == 240:
        output.index = pd.to_datetime(output.index.date)

    return output


def index_futures_adj(frequency=1):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    path = os.path.join(dirname, 'dataset', '_adj_close_price.csv')
    output = pd.read_csv(path, index_col=0, header=0)
    output.index = pd.to_datetime(output.index)
    if 240 % frequency != 0:
        print(F'240%frequency should be 0')
        return

    if frequency <= 240:
        output = output[frequency-1::frequency]
    elif frequency == 240:
        output = output[239::240]
        output.index = output.index.to_series().apply(lambda x: datetime.datetime(x.year, x.month, x.day))

    return output


def index_futures_volume(frequency=1):
    dirname, _ = os.path.split(os.path.abspath(__file__))
    path = os.path.join(dirname, 'dataset', '_volume.csv')
    output = pd.read_csv(path, index_col=0, header=0)
    output.index = pd.to_datetime(output.index)
    if 240 % frequency != 0:
        print(F'240%frequency should be 0')
        return

    if frequency <= 240:
        output = output.rolling(frequency).sum().iloc[frequency-1::frequency]
    elif frequency == 240:
        output = output.groupby(output.index.date).sum()
        output.index = pd.to_datetime(output.index)

    return output
