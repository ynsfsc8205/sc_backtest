import pandas as pd
import numpy as np
from copy import copy


def _preprocessing(x, y):
    if len(x) != len(y):
        print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
        return

    x = copy(x)
    y = copy(y)

    factor = pd.Series(x)
    ret = pd.Series(y)

    valid_x = np.isfinite(factor)
    factor = factor[valid_x]
    ret = ret[valid_x]

    factor.dropna(inplace=True)
    ret = ret.reindex(factor.index)

    return factor, ret


def _preprocessing_cs(x, y):
    x = copy(x)
    y = copy(y)

    factor = pd.DataFrame(x)
    ret = pd.DataFrame(y)

    factor[np.isinf(factor)] = np.nan
    ret[np.isinf(ret)] = np.nan

    return factor, ret


def _get_report(delta, ret):
    """
    :param delta: pd.Series
    :param ret: pd.Series
    :return: report: pd.Series
    """
    if len(delta) != len(ret):
        print(F'mismatched length of indicator, ret, len(indicator) == {len(delta)} len(ret) == {len(ret)}')
        return

    Index = ['PnL', 'PoT', 'Sharpe', 'MDD', 'MDDD', 'Calmar', 'HP',
             'WR', 'Odds', 'Worst', 'MCW', 'MCL', 'Skewness', 'Kurtosis']

    try:
        pnl = delta * ret

        day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()

        # year PnL
        pnl_y = round(day_pnl.mean() * 250, 2)

        # PoT
        pot = round(pnl.sum() / (delta.diff().abs().sum()) * 1e4, 2)

        # Sharpe
        sharpe = round(day_pnl.mean() / day_pnl.std() * np.sqrt(250), 2)

        # MDD
        DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()
        MDD = round(DD_ss.abs().max(), 2)

        def _calc_mddd(m):
            count = 0
            for i in range(-1, -len(m) - 1, -1):
                if m.iloc[i] < 0:
                    count += 1
                else:
                    break
            return count

        MDDD = int(DD_ss.expanding(0).apply(_calc_mddd).max())

        # HP
        HP = 2*delta.abs().mean()/delta.diff().abs().mean()

        # WR
        WR = round(day_pnl[day_pnl > 0].count() / len(day_pnl[day_pnl != 0]), 4)

        # Odds
        Odds = day_pnl[day_pnl > 0].mean() / day_pnl[day_pnl < 0].abs().mean()

        # Worst
        Worst = round(day_pnl.min(), 2)

        # MCW, MCL
        temp = np.sign(day_pnl)
        temp = temp.groupby((temp != temp.shift()).cumsum()).cumsum()
        MCW = temp.max()
        MCL = -temp.min()

        # Skewness
        Skew = round(day_pnl.skew(), 4)

        # Kurtosis
        Kurt = round(day_pnl.kurtosis(), 4)

        output = pd.Series([pnl_y, pot, sharpe, MDD, MDDD, round(pnl_y/MDD, 2),
                            HP, WR, Odds, Worst, MCW, MCL, Skew, Kurt], index=Index)

    except:
        output = pd.Series(np.nan, index=Index)

    return output


def _get_report2(pnl):
    Index = ['PnL', 'Sharpe', 'MDD', 'MDDD', 'Calmar',
             'WR', 'Odds', 'Worst', 'MCW', 'MCL', 'Skewness', 'Kurtosis']

    try:
        day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()

        # year PnL
        pnl_y = round(day_pnl.mean() * 250, 2)

        # Sharpe
        sharpe = round(day_pnl.mean() / day_pnl.std() * np.sqrt(250), 2)

        # MDD
        DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()
        MDD = round(DD_ss.abs().max(), 2)

        def _calc_mddd(m):
            count = 0
            for i in range(-1, -len(m) - 1, -1):
                if m.iloc[i] < 0:
                    count += 1
                else:
                    break
            return count

        MDDD = int(DD_ss.expanding(0).apply(_calc_mddd).max())

        # WR
        WR = round(day_pnl[day_pnl > 0].count() / len(day_pnl[day_pnl != 0]), 4)

        # Odds
        Odds = day_pnl[day_pnl > 0].mean() / day_pnl[day_pnl < 0].abs().mean()

        # Worst
        Worst = round(day_pnl.min(), 2)

        # MCW, MCL
        temp = np.sign(day_pnl)
        temp = temp.groupby((temp != temp.shift()).cumsum()).cumsum()
        MCW = temp.max()
        MCL = -temp.min()

        # Skewness
        Skew = round(day_pnl.skew(), 4)

        # Kurtosis
        Kurt = round(day_pnl.kurtosis(), 4)

        output = pd.Series([pnl_y, sharpe, MDD, MDDD, round(pnl_y/MDD, 2),
                            WR, Odds, Worst, MCW, MCL, Skew, Kurt], index=Index)

    except:
        output = pd.Series(np.nan, index=Index)

    return output


def _get_report_composite(delta, ret):
    """
    :param delta: report pd.DataFrame
    :param ret: report pd.DataFrame
    :return: report pd.DataFrame
    """
    if len(delta) != len(ret):
        print(F'mismatched length of indicator, ret, len(indicator) == {len(delta)} len(ret) == {len(ret)}')
        return

    Index = ['PnL', 'PoT', 'Sharpe', 'MDD', 'MDDD', 'Calmar', 'HP',
             'WR', 'Odds', 'Worst', 'MCW', 'MCL', 'Skewness', 'Kurtosis']

    try:
        pnl = (delta * ret).sum(axis=1)

        day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()

        # year PnL
        pnl_y = round(day_pnl.mean() * 250, 2)

        # PoT
        pot = round(pnl.sum() / (delta.diff().abs().sum().sum()) * 1e4, 2)

        # Sharpe
        sharpe = round(day_pnl.mean() / day_pnl.std() * np.sqrt(250), 2)

        # MDD
        DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()
        MDD = round(DD_ss.abs().max(), 2)

        def _calc_mddd(m):
            count = 0
            for i in range(-1, -len(m) - 1, -1):
                if m.iloc[i] < 0:
                    count += 1
                else:
                    break
            return count

        MDDD = int(DD_ss.expanding(0).apply(_calc_mddd).max())

        # HP
        HP = 2*delta.abs().sum(axis=1).mean()/delta.diff().abs().sum(axis=1).mean()

        # WR
        WR = round(day_pnl[day_pnl > 0].count() / len(day_pnl[day_pnl != 0]), 4)

        # Odds
        Odds = day_pnl[day_pnl > 0].mean() / day_pnl[day_pnl < 0].abs().mean()

        # Worst
        Worst = round(day_pnl.min(), 2)

        # MCW, MCL
        temp = np.sign(day_pnl)
        temp = temp.groupby((temp != temp.shift()).cumsum()).cumsum()
        MCW = temp.max()
        MCL = -temp.min()

        # Skewness
        Skew = round(day_pnl.skew(), 4)

        # Kurtosis
        Kurt = round(day_pnl.kurtosis(), 4)

        output = pd.Series([pnl_y, pot, sharpe, MDD, MDDD, round(pnl_y/MDD, 2),
                            HP, WR, Odds, Worst, MCW, MCL, Skew, Kurt], index=Index)

    except:
        output = pd.Series(0, index=Index)

    return output


def _get_report_composite2(pnl):
    Index = ['PnL', 'Sharpe', 'MDD', 'MDDD', 'Calmar',
             'WR', 'Odds', 'Worst', 'MCW', 'MCL', 'Skewness', 'Kurtosis']

    try:
        day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda x: x.date())).sum()

        # year PnL
        pnl_y = round(day_pnl.mean() * 250, 2)

        # Sharpe
        sharpe = round(day_pnl.mean() / day_pnl.std() * np.sqrt(250), 2)

        # MDD
        DD_ss = day_pnl.cumsum() - day_pnl.cumsum().expanding(0).max()
        MDD = round(DD_ss.abs().max(), 2)

        def _calc_mddd(m):
            count = 0
            for i in range(-1, -len(m) - 1, -1):
                if m.iloc[i] < 0:
                    count += 1
                else:
                    break
            return count

        # MDDD = np.sign(DD_ss.abs()).cumsum()
        # MDDD = MDDD[MDDD.diff() == 0].diff().max()
        MDDD = int(DD_ss.expanding(0).apply(_calc_mddd).max())

        # WR
        WR = round(day_pnl[day_pnl > 0].count() / len(day_pnl[day_pnl != 0]), 4)

        # Odds
        Odds = day_pnl[day_pnl > 0].mean() / day_pnl[day_pnl < 0].abs().mean()

        # Worst
        Worst = round(day_pnl.min(), 2)

        # MCW, MCL
        temp = np.sign(day_pnl)
        temp = temp.groupby((temp != temp.shift()).cumsum()).cumsum()
        MCW = temp.max()
        MCL = -temp.min()

        # Skewness
        Skew = round(day_pnl.skew(), 4)

        # Kurtosis
        Kurt = round(day_pnl.kurtosis(), 4)

        output = pd.Series([pnl_y, sharpe, MDD, MDDD, round(pnl_y/MDD, 2),
                            WR, Odds, Worst, MCW, MCL, Skew, Kurt], index=Index)

    except:
        output = pd.Series(0, index=Index)

    return output


def _corr2(x, y):
    sorted_idx = np.argsort(x)
    cdf_line = np.cumsum(y.iloc[sorted_idx] - np.nanmean(y))
    output = -np.sign(cdf_line.sum())
    return output
