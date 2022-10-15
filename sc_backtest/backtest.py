import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .private_func import _get_report, _get_report_composite, _get_report2, _get_report_composite2
import warnings
import datetime
import chinese_calendar as cc
from .ta_personal import is_trading_day, not_trading_day

warnings.filterwarnings("ignore")


def get_report(delta, ret, long_short_split=True):
    """
    delta: pd.Series with DatatimeIndex
    ret: pd.Series with DatatimeIndex
    report: pd.DataFrame
    """
    if not (isinstance(delta, pd.Series)) or not (isinstance(ret, pd.Series)):
        print('indicator and pnl must be pd.Series')
        return

    if not (isinstance(delta.index, pd.core.indexes.datetimes.DatetimeIndex)) \
            or not (isinstance(ret.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of indicator and pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    if len(delta) < len(ret):
        ret = ret.copy().reindex(delta.index)
        indicator = delta.copy()
    else:
        indicator = delta.copy().reindex(ret.index)
        ret = ret.copy()

    valid_ind = np.isfinite(indicator)
    indicator = indicator[valid_ind]
    ret = ret[valid_ind]

    report_all = _get_report(indicator, ret).rename('All').to_frame().T
    report_sub = indicator.groupby(by=indicator.index.to_series().apply(lambda x: x.year)) \
        .apply(lambda x: _get_report(x, ret.reindex(x.index))).unstack()

    if not long_short_split:
        return pd.concat([report_all, report_sub])
    else:
        indicator_long = indicator.copy()
        indicator_long[indicator_long < 0] = 0
        report_long = _get_report(indicator_long, ret).rename('Long').to_frame().T

        indicator_short = indicator.copy()
        indicator_short[indicator_long > 0] = 0
        report_short = _get_report(indicator_short, ret).rename('Short').to_frame().T

        return pd.concat([report_all, report_long, report_short, report_sub])


def get_report2(pnl):
    """
    pnl: pd.Series with DatatimeIndex
    """
    if not (isinstance(pnl, pd.Series)):
        print('indicator and pnl must be pd.Series')
        return

    if not (isinstance(pnl.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of indicator and pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    report_all = _get_report2(pnl).rename('All').to_frame().T
    report_sub = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.year)) \
        .apply(lambda x: _get_report2(x)).unstack()

    return pd.concat([report_all, report_sub])


def get_composite_report(delta, ret, long_short_split=True):
    """
    delta: pd.DataFrame with DatatimeIndex
    ret: pd.DataFrame with DatatimeIndex
    report: pd.DataFrame
    """
    if not (isinstance(delta, pd.DataFrame)) or not (isinstance(ret, pd.DataFrame)):
        print('indicator and pnl must be pd.Series')
        return

    if not (isinstance(delta.index, pd.core.indexes.datetimes.DatetimeIndex)) \
            or not (isinstance(ret.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of indicator and pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    if len(delta) < len(ret):
        ret = ret.copy().reindex(delta.index)
        indicator = delta.copy()
    else:
        indicator = delta.copy().reindex(ret.index)
        ret = ret.copy()

    inf_ind = np.isinf(indicator)
    indicator[inf_ind] = np.nan
    indicator.dropna(inplace=True)
    ret = ret.reindex(indicator.index)

    report_all = _get_report_composite(indicator, ret).rename('All').to_frame().T
    # report_sub = indicator.groupby(by=indicator.index.year) \
    #     .apply(lambda x: _get_report_composite(x, ret.reindex(x.index))).unstack()
    report_sub = indicator.groupby(by=indicator.index.year) \
        .apply(lambda x: _get_report_composite(x, ret.reindex(x.index)))

    if not long_short_split:
        return pd.concat([report_all, report_sub])
    else:
        indicator_long = indicator.copy()
        indicator_long[indicator_long < 0] = 0
        report_long = _get_report_composite(indicator_long, ret).rename('Long').to_frame().T

        indicator_short = indicator.copy()
        indicator_short[indicator_long > 0] = 0
        report_short = _get_report_composite(indicator_short, ret).rename('Short').to_frame().T

        return pd.concat([report_all, report_long, report_short, report_sub])


def get_pnl_plot(delta, ret, fig_return=False, alpha=True, **kwargs):
    """
    delta: pd.Series with DatatimeIndex
    ret: pd.Series with DatatimeIndex
    """
    if not (isinstance(delta, pd.Series)) or not (isinstance(ret, pd.Series)):
        print('indicator and pnl must be pd.Series')
        return

    if not (isinstance(delta.index, pd.core.indexes.datetimes.DatetimeIndex)) \
            or not (isinstance(ret.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of indicator and pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    if len(delta) < len(ret):
        ret = ret.copy().reindex(delta.index)
        indicator = delta.copy()
    else:
        indicator = delta.copy().reindex(ret.index)
        ret = ret.copy()

    valid_ind = np.isfinite(indicator)
    indicator = indicator[valid_ind]
    ret = ret[valid_ind]

    if 'drop_holiday' in kwargs:
        dh = indicator.index.to_series().apply(lambda d: 0.0 if all([not_trading_day(d + datetime.timedelta(days=i))
                                                                     for i in range(1, 4)]) else 1.0).to_numpy()
        indicator = indicator * dh

    report_all = _get_report(indicator, ret)

    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    pnl = indicator * ret
    ind_long = indicator[indicator > 0]
    ind_short = indicator[indicator <= 0]
    pnl_long = ind_long * ret.reindex(ind_long.index)
    pnl_short = ind_short * ret.reindex(ind_short.index)

    day_ret = (ret * 100).groupby(by=ret.index.to_series().apply(lambda x: x.date())).sum()
    day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()
    day_pnl_long = pnl_long.groupby(by=pnl_long.index.to_series().apply(lambda x: x.date())).sum()
    day_pnl_short = pnl_short.groupby(by=pnl_short.index.to_series().apply(lambda x: x.date())).sum()
    DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()

    #
    if 'title' in kwargs:
        temp = kwargs.pop('title')
        s_label = f'Strategy: {temp}'
    else:
        s_label = 'Strategy'

    day_pnl.cumsum().plot(ax=ax1, label=s_label)
    day_pnl_long.cumsum().plot(ax=ax1, label='Only Long', c='r', ls='--')
    day_pnl_short.cumsum().plot(ax=ax1, label='Only Short', c='g', ls='--')
    day_ret.cumsum().plot(ax=ax1, ls=':', c='k', alpha=0.5, label='Asset')

    if alpha:
        print(_get_report2(day_pnl - day_ret).rename('Alpha Report').to_frame().T)
        (day_pnl.cumsum() - day_ret.cumsum()).plot(ax=ax1, label='Alpha', ls='--', c='y')

    ax1.set_title('PnL={} PoT={} Sharpe={} Calmar={} WR={:.2f} HP={:.2f}'.format(
        report_all.loc['PnL'], report_all.loc['PoT'], report_all.loc['Sharpe'],
        report_all.loc['Calmar'], report_all.loc['WR'], report_all.loc['HP']
    ))
    ax1.set_ylabel('PnL')
    ax1.legend(loc=2)
    ax1.set_xlabel('')
    ax1.grid()

    DD_ss.plot(ax=ax2, c='r')
    ax2.set_title('MDD={} MDDD={} Worst={:.2f}'.format(
        report_all.loc['MDD'], report_all.loc['MDDD'], report_all.loc['Worst']
    ))
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid()

    plt.show()
    if fig_return:
        return fig

    return


def get_pnl_plot2(pnl, fig_return=False, **kwargs):
    """
    pnl: pd.Series with DatatimeIndex
    """
    if not (isinstance(pnl, pd.Series)):
        print('and pnl must be pd.Series')
        return

    if not (isinstance(pnl.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    report_all = _get_report2(pnl)

    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    # pnl

    day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()
    DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()

    day_pnl.cumsum().plot(ax=ax1, label='Strategy')

    ax1.set_title('PnL={} Sharpe={} Calmar={} WR={:.2f}'.format(
        report_all.loc['PnL'], report_all.loc['Sharpe'],
        report_all.loc['Calmar'], report_all.loc['WR']
    ))
    ax1.set_ylabel('PnL')
    ax1.legend(loc=2)
    ax1.set_xlabel('')
    ax1.grid()

    DD_ss.plot(ax=ax2, c='r')
    ax2.set_title('MDD={} MDDD={} Worst={:.2f}'.format(
        report_all.loc['MDD'], report_all.loc['MDDD'], report_all.loc['Worst']
    ))
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid()

    plt.show()
    if fig_return:
        return fig

    return


def get_composite_pnl_plot(delta, ret, fig_return=False, alpha=True,
                           long_short=False, benchmark_drawdown=False, **kwargs):
    """
    Multi-Assets PnL
    delta: pd.DataFrame with DatatimeIndex
    ret: pd.DataFrame with DatatimeIndex
    """
    if not (isinstance(delta, pd.DataFrame)) or not (isinstance(ret, pd.DataFrame)):
        print('indicator and pnl must be pd.DataFrame')
        return

    if not (isinstance(delta.index, pd.core.indexes.datetimes.DatetimeIndex)) \
            or not (isinstance(ret.index, pd.core.indexes.datetimes.DatetimeIndex)):
        print('index of indicator and pnl must be pd.core.indexes.datetimes.DatetimeIndex')
        return

    if len(delta) < len(ret):
        ret = ret.copy().reindex(delta.index)
        indicator = delta.copy()
    else:
        indicator = delta.copy().reindex(ret.index)
        ret = ret.copy()

    inf_ind = np.isinf(indicator)
    indicator[inf_ind] = np.nan
    indicator.fillna(0, inplace=True)

    if 'drop_holiday' in kwargs:
        dh = indicator.index.to_series().apply(lambda d: 0.0 if all([not_trading_day(d + datetime.timedelta(days=i))
                                                                     for i in range(1, 4)]) else 1.0).to_numpy()
        indicator = indicator.multiply(dh, axis=0)

    # indicator_og = pd.DataFrame(100.0 / len(indicator.columns), index=indicator.index, columns=indicator.columns)
    if 'benchmark' in kwargs:
        indicator_og = kwargs.pop('benchmark')
    else:
        indicator_og = pd.DataFrame(np.nan, index=indicator.index, columns=indicator.columns)
        indicator_og[~ np.isnan(ret)] = 100
        indicator_og = indicator_og.div(indicator_og.count(axis=1), axis=0)

    # plot
    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    #
    pnl = (indicator * ret).sum(axis=1)

    #
    day_ret = (indicator_og * ret).sum(axis=1).groupby(by=ret.index.to_series().apply(lambda x: x.date())).sum()
    day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()
    DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()


    #
    if 'title' in kwargs:
        temp = kwargs.pop('title')
        s_label = f'Strategy: {temp}'
    else:
        s_label = 'Strategy'

    day_pnl.cumsum().plot(ax=ax1, label=s_label)
    #
    if long_short:
        ind_long = indicator.copy()
        ind_short = indicator.copy()
        ind_long[ind_long < 0] = 0
        ind_short[ind_short > 0] = 0
        pnl_long = (ind_long * ret).sum(axis=1)
        pnl_short = (ind_short * ret).sum(axis=1)
        day_pnl_long = pnl_long.groupby(by=pnl_long.index.to_series().apply(lambda x: x.date())).sum()
        day_pnl_short = pnl_short.groupby(by=pnl_short.index.to_series().apply(lambda x: x.date())).sum()
        day_pnl_long.cumsum().plot(ax=ax1, label='Only Long', c='r', ls='--')
        day_pnl_short.cumsum().plot(ax=ax1, label='Only Short', c='g', ls='--')
    #
    day_ret.cumsum().plot(ax=ax1, ls=':', c='k', alpha=0.5, label='Asset')
    if alpha:
        print(_get_report_composite2(pnl - (indicator_og * ret).sum(axis=1)).rename('Alpha Report').to_frame().T)
        (day_pnl.cumsum() - day_ret.cumsum()).plot(ax=ax1, label='Alpha', ls='--', c='y')

    report_all = _get_report_composite(indicator, ret)

    ax1.set_title('PnL={} PoT={} Sharpe={} Calmar={} WR={:.2f}  HP={:.2f}'.format(
        report_all.loc['PnL'], report_all.loc['PoT'], report_all.loc['Sharpe'],
        report_all.loc['Calmar'], report_all.loc['WR'], report_all.loc['HP']
    ))
    ax1.set_ylabel('PnL')
    ax1.legend(loc=2)
    ax1.set_xlabel('')
    ax1.grid()

    DD_ss.plot(ax=ax2, c='r', label=s_label)
    if benchmark_drawdown:
        DD_og_ss = day_ret.cumsum() - day_ret.cumsum().expanding(0).max()
        DD_og_ss.plot(ax=ax2, c='k', ls=':', alpha=0.5, label='Asset')

    ax2.set_title('MDD={} MDDD={} Worst={:.2f}'.format(
        report_all.loc['MDD'], report_all.loc['MDDD'], report_all.loc['Worst']
    ))
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend(loc=2)
    ax2.grid()

    plt.show()
    if fig_return:
        return fig

    return


def round_test(delta, close, adj_close, BPV, mult=1e4, fee_rate=3e-4, vol_limit=1):
    if not isinstance(delta, pd.DataFrame):
        print('delta must be pd.DataFrame')
        return

    if not isinstance(close, pd.DataFrame):
        print('close must be pd.DataFrame')
        return

    if not isinstance(adj_close, pd.DataFrame):
        print('adj_close must be pd.DataFrame')
        return

    if not isinstance(BPV, pd.DataFrame):
        print('BPV must be pd.DataFrame')
        return

    if not isinstance(vol_limit, int):
        print('vol_limit must be int')
        return

    if not all([set(delta.columns) == set(close.columns),
                set(adj_close.columns) == set(close.columns),
                set(BPV.columns) == set(close.columns)]):
        print(F'mismatched columns: delta{delta.columns}, close{close.columns}, \
              adj_close{adj_close.columns}, BPV{BPV.columns}')
        return

    if not all([len(delta) == len(close), len(close) == len(adj_close), len(adj_close) == len(BPV)]):
        print(F'mismatched length: delta{len(delta)}, close{len(close)}, \
              adj_close{len(adj_close)}, BPV{len(BPV)}')
        return

    # vol without round
    og_vol = ((delta * mult) / (BPV * close)).fillna(0)

    # rounded vol
    vol = np.sign(og_vol) * og_vol.abs().round()

    # vol_limit
    def vol_limit_apply(x):
        output = 0
        if x >= vol_limit:
            output = vol_limit
        elif x <= -vol_limit:
            output = -vol_limit
        else:
            output = round(x)

        return output

    new_vol = og_vol.applymap(vol_limit_apply)

    #
    ret = adj_close.pct_change().shift(-1).fillna(0)

    #
    og_pnl = delta * mult * ret
    og_day_pnl = og_pnl.groupby(by=og_pnl.index.date).sum().sum(axis=1)
    # og_sum_pnl = og_pnl.cumsum().dropna().sum(axis=1)
    og_sum_pnl = og_pnl.cumsum().sum(axis=1)
    og_y_pnl = round(og_day_pnl.mean() / mult * 251, 2)

    pnl = vol * BPV * close * ret
    day_pnl = pnl.groupby(by=og_pnl.index.date).sum().sum(axis=1)
    # sum_pnl = pnl.cumsum().dropna().sum(axis=1)
    sum_pnl = pnl.cumsum().sum(axis=1)
    y_pnl = round(day_pnl.mean() / mult * 251, 2)

    # after cost
    sum_pnl_ac = (pnl - (vol.diff() * BPV * close).abs() * fee_rate).cumsum().sum(axis=1)
    day_pnl_ac = (pnl - (vol.diff() * BPV * close).abs() * fee_rate) \
        .groupby(by=og_pnl.index.date).sum().sum(axis=1)
    y_pnl_ac = round(day_pnl_ac.mean() / mult * 251, 2)

    new_pnl = new_vol * BPV * close * ret
    # new_sum_pnl = new_pnl.dropna().cumsum().sum(axis=1)
    new_sum_pnl = new_pnl.cumsum().sum(axis=1)
    new_day_pnl = new_pnl.groupby(by=og_pnl.index.date).sum().sum(axis=1)
    new_y_pnl = round(new_day_pnl.mean() / mult * 251, 2)

    # new after cost
    new_sum_pnl_ac = (new_pnl - (new_vol.diff() * BPV * close).abs() * fee_rate).cumsum().sum(axis=1)
    new_day_pnl_ac = (new_pnl - (new_vol.diff() * BPV * close).abs() * fee_rate) \
        .groupby(by=og_pnl.index.date).sum().sum(axis=1)
    new_y_pnl_ac = round(new_day_pnl_ac.mean() / mult * 251, 2)

    #
    og_pot = (og_pnl.sum().sum() / (og_vol.diff() * BPV * close).abs().sum().sum()) * 1e4
    pot = (pnl.sum().sum() / (vol.diff() * BPV * close).abs().sum().sum()) * 1e4
    new_pot = (new_pnl.sum().sum() / (new_vol.diff() * BPV * close).abs().sum().sum()) * 1e4

    def _sharpe(m):
        return m.mean() / m.std() * np.sqrt(250)

    data_pnl = pd.concat([
        og_sum_pnl.rename('Orginal Delta: PnL={} Sharpe={:.2f} PoT={:.2f}'. \
                          format(og_y_pnl, _sharpe(og_day_pnl), og_pot)),
        sum_pnl.rename('Round Volume: PnL={} Sharpe={:.2f} PoT={:.2f}'. \
                       format(y_pnl, _sharpe(day_pnl), pot)),
        new_sum_pnl.rename('{} Volume Limit: PnL={} Sharpe={:.2f} PoT={:.2f}'. \
                           format(int(vol_limit), new_y_pnl, _sharpe(new_day_pnl), new_pot)),
        sum_pnl_ac.rename('Round Volume after-cost: PnL={} Sharpe={:.2f}'. \
                          format(y_pnl_ac, _sharpe(day_pnl_ac))),
        new_sum_pnl_ac.rename('{} Volume Limit after-cost: PnL={} Sharpe={:.2f}'. \
                              format(int(vol_limit), new_y_pnl_ac,
                                     _sharpe(new_day_pnl_ac)))], axis=1) / mult

    # plot
    fig = plt.figure(figsize=(20, 16))

    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax3 = plt.subplot2grid((4, 1), (3, 0))

    sns.lineplot(data=data_pnl, ax=ax1)
    ax1.set_title('Round-Test: mult={}k vol_limit={} fee_rate={}'.format(int(mult / 1e3),
                                                                         int(vol_limit),
                                                                         str(fee_rate * 100) + '%'))
    ax1.set_xlabel('')
    ax1.set_ylabel('PnL')

    sns.lineplot(data=vol, ax=ax2)
    ax2.set_xlabel('')
    ax2.set_ylabel('Volume')

    sns.lineplot(data=np.sign(new_vol.dropna()).astype(int), ax=ax3)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Long or Short')
    ax3.set_yticks([-1, 0, 1])

    plt.show()

    return fig
