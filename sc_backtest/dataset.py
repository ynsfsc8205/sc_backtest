import pandas as pd
import os


def get_data(data_name='adj_close_price',
             columns=None,
             start_time='2016-01-01', end_time='2021-01-01',
             frequency=1):
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
    output = pd.read_csv(path, index_col=0, header=0)
    output.index = pd.to_datetime(output.index)
    output = output.loc[pd.to_datetime(start_time):pd.to_datetime(end_time), columns]

    if frequency <= 240 and frequency != 1:
        if data_name in ['adj_close_price', 'openint']:
            output = output.resample(f'{int(frequency)}T', label='right', closed='right').last().dropna()
        elif data_name in ['adj_open_price']:
            output = output.resample(f'{int(frequency)}T', label='right', closed='right').first().dropna()
        elif data_name in ['adj_high_price']:
            output = output.resample(f'{int(frequency)}T', label='right', closed='right').max().dropna()
        elif data_name in ['adj_low_price']:
            output = output.resample(f'{int(frequency)}T', label='right', closed='right').min().dropna()
        elif data_name in ['volume', 'value']:
            output = output.resample(f'{int(frequency)}T', label='right', closed='right').sum().dropna()

    if frequency == 240:
        output.index = pd.to_datetime(output.index.date)

    return output
