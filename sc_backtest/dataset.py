import numpy as np
import pandas as pd
import os
import datetime


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

    if frequency <= 120 and frequency != 1:
        output_mn = output.copy()
        output_mn[output_mn.index.time > datetime.time(12, 0)] = np.nan
        output_mn.dropna(inplace=True)
        output_af = output.copy()
        output_af[output_af.index.time < datetime.time(12, 0)] = np.nan
        output_af.dropna(inplace=True)
        if data_name in ['adj_close_price', 'openint']:
            output_mn = output_mn.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_mn.index[frequency-1]).last(min_count=1).dropna()
            output_af = output_af.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_af.index[frequency-1]).last(min_count=1).dropna()
        elif data_name in ['adj_open_price']:
            output_mn = output_mn.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_mn.index[frequency-1]).first(min_count=1).dropna()
            output_af = output_af.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_af.index[frequency-1]).first(min_count=1).dropna()
        elif data_name in ['adj_high_price']:
            output_mn = output_mn.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_mn.index[frequency-1]).max(min_count=1).dropna()
            output_af = output_af.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_af.index[frequency-1]).max(min_count=1).dropna()
        elif data_name in ['adj_low_price']:
            output_mn = output_mn.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_mn.index[frequency-1]).min(min_count=1).dropna()
            output_af = output_af.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_af.index[frequency-1]).min(min_count=1).dropna()
        elif data_name in ['volume', 'value']:
            output_mn = output_mn.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_mn.index[frequency-1]).sum(min_count=1).dropna()
            output_af = output_af.resample(f'{int(frequency)}T',
                                           label='right',
                                           closed='right',
                                           origin=output_af.index[frequency-1]).sum(min_count=1).dropna()
        output = pd.concat([output_mn, output_af], axis=0).sort_index()

    if frequency == 240:
        if data_name in ['adj_close_price', 'openint']:
            output = output.resample('1d', closed='right').last(min_count=1).dropna()
        elif data_name in ['adj_open_price']:
            output = output.resample('1d', closed='right').first(min_count=1).dropna()
        elif data_name in ['adj_high_price']:
            output = output.resample('1d', closed='right').max(min_count=1).dropna()
        elif data_name in ['adj_low_price']:
            output = output.resample('1d', closed='right').min(min_count=1).dropna()
        elif data_name in ['volume', 'value']:
            output = output.resample('1d', closed='right').sum(min_count=1).dropna()
        output.index = pd.to_datetime(output.index)

    return output
