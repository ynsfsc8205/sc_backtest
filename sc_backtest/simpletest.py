# -*- coding: utf-8 -*-
"""
@author: chang.sun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .private_func import _preprocessing, _corr2, _preprocessing_cs
import warnings
from .backtest import get_pnl_plot

warnings.filterwarnings("ignore")


class simpletest():
    """
    Index Future Back_test Module
    """

    def __init__(self):
        pass

    def get_cdf_peak(self, x, y):
        """
        x: np.nddary or pd.Series
        y: np.nddary or pd.Series
        """
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        corr = _corr2(x, y)
        x_, y_ = _preprocessing(x, y)

        sorted_idx = np.argsort(x_)
        cdf_line = np.cumsum(y_.iloc[sorted_idx] - np.nanmean(y_))

        peak_idx = np.argmin(cdf_line) if corr > 0 else np.argmax(cdf_line)
        peak = x_[sorted_idx[peak_idx]]
        return peak

    def simple_pnl(self, x, y,
                   ax=None, st_plot=True, data_return=True,
                   sign=True, val=True, peak=True,
                   bt_plot=False, **kwargs):
        """
        x: pd.Series
        y: pd.Series
        Sign-Trade: factor >(<) 0 -> indicator = 1(-1) * sign(corr)
        Val-Trade: indicator =  factor/factor.abs.mean * 100
        Peak-Trade: factor >(<) cdf_peak_factor_val -> indicator = 1(-1) * sign(corr)
        """

        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if not (isinstance(x, pd.Series)) or not (isinstance(y, pd.Series)):
            print('x and y must be pd.Series')
            return

        if not (isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex)) \
                or not (isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex)):
            print('index of x and y must be pd.core.indexes.datetimes.DatetimeIndex')
            return

        factor, ret = _preprocessing(x, y)

        corr = _corr2(factor, ret)

        output = {}

        if st_plot:
            if ax is None:
                fig = plt.figure(figsize=(15, 6))
                ax = fig.add_subplot(111)

        if sign:
            indicator1 = np.sign(factor) * 100 * np.sign(corr)
            label1 = 'Sign-Trade: '

            pnl1 = ret * indicator1
            sum_pnl1 = pnl1.cumsum()

            day_pnl1 = pnl1.groupby(by=pnl1.index.to_series().apply(lambda m: m.date())).sum()

            # PoT
            pot1 = pnl1.sum() / (indicator1.diff().abs().sum()) * 1e4

            # Sharpe
            sharpe1 = day_pnl1.mean() / day_pnl1.std() * 16

            # year PnL
            pnl_y1 = day_pnl1.mean() * 251

            if st_plot:
                sum_pnl1.plot(ax=ax, label=label1 +
                                           'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                              .format(pnl_y1, sharpe1, pot1))

            if data_return:
                output['pnl_sign'] = sum_pnl1
                output['delta_sign'] = indicator1

        if val:
            indicator2 = factor / factor.std() * 100 * np.sign(corr)
            label2 = 'Val-Trade: '

            pnl2 = ret * indicator2
            sum_pnl2 = pnl2.cumsum()

            day_pnl2 = pnl2.groupby(by=pnl2.index.to_series().apply(lambda m: m.date())).sum()

            # PoT
            pot2 = pnl2.sum() / (indicator2.diff().abs().sum()) * 1e4

            # Sharpe
            sharpe2 = day_pnl2.mean() / day_pnl2.std() * 16

            # year PnL
            pnl_y2 = day_pnl2.mean() * 251

            if st_plot:
                sum_pnl2.plot(ax=ax, label=label2 +
                                           'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                              .format(pnl_y2, sharpe2, pot2))

            if data_return:
                output['pnl_val'] = sum_pnl2
                output['delta_val'] = indicator2

        if peak:
            peak_val = self.get_cdf_peak(factor, ret)
            indicator3 = factor.apply(lambda m: 1 if m >= peak_val else -1) * 100 * np.sign(corr)
            label3 = 'Peak-Trade: '

            pnl3 = ret * indicator3
            sum_pnl3 = pnl3.cumsum()

            day_pnl3 = pnl3.groupby(by=pnl3.index.to_series().apply(lambda m: m.date())).sum()

            # PoT
            pot3 = pnl3.sum() / (indicator3.diff().abs().sum()) * 1e4

            # Sharpe
            sharpe3 = day_pnl3.mean() / day_pnl3.std() * 16

            # year PnL
            pnl_y3 = day_pnl3.mean() * 251

            if st_plot:
                sum_pnl3.plot(ax=ax, label=label3 +
                                           'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                              .format(pnl_y3, sharpe3, pot3))

            if data_return:
                output['pnl_peak'] = sum_pnl3
                output['delta_peak'] = indicator3

        if st_plot:
            if ax is None:
                ax = plt.gca()
            (y.cumsum() * 100.0).dropna().plot(ax=ax,
                                               label='Asset Cumulative Return', ls=':', c='k',
                                               alpha=0.5)
            if 'title' in kwargs:
                title = kwargs.pop('title')
                ax.set_title(f'Simple Time-Series Test (info: {str(title)})')
            else:
                ax.set_title('Simple Time-Series Test')
            ax.set_ylabel('PnL')
            ax.set_xlabel('Date')
            ax.legend()

        # back_test pnl plot
        if bt_plot:
            if sign:
                get_pnl_plot(indicator1, ret, **kwargs)

            if val:
                get_pnl_plot(indicator2, ret, **kwargs)

            if peak:
                get_pnl_plot(indicator3, ret, **kwargs)

        if data_return:
            return output

        return

    def threshold_pnl(self, x, y, t_list=None, sign_list=None,
                      ax=None, st_plot=False, data_return=True,
                      bt_plot=True, **kwargs):
        """
        x: pd.Series
        y: pd.Series
        t_list: list: contains i thresold(s) where i >= 1
        sign_list: list: contains i+1 signs(1, 0, -1)
        """

        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if not (isinstance(x, pd.Series)) or not (isinstance(y, pd.Series)):
            print('x and y must be pd.Series')
            return

        if not (isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex)) \
                or not (isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex)):
            print('index of x and y must be pd.core.indexes.datetimes.DatetimeIndex')
            return

        if sign_list is None:
            sign_list = [1, 1]
        if t_list is None:
            t_list = [x[0]]

        if len(sign_list) - len(t_list) != 1:
            print(F'mismatched length of t_list and sign_list len(t_list) == {len(t_list)} \
            len(sign_list) == {len(sign_list)} (len(sign_list)-len(t_list) should be 1)')
            return

        if len(t_list) == 0:
            print(F'len(t_list) must be greater than or equal to 1')
            return

        factor, ret = _preprocessing(x, y)
        indicator = factor.copy()

        t_min = factor.min()
        t_max = factor.max()

        for i in range(len(sign_list)):
            if i == 0:
                t_left = t_min
                t_right = t_list[i]
            elif i == len(sign_list) - 1:
                t_left = t_list[i - 1]
                t_right = t_max
            else:
                t_left = t_list[i - 1]
                t_right = t_list[i]
            indicator.loc[(factor >= t_left) & (factor < t_right)] = sign_list[i] * 100.0

        pnl = ret * indicator
        sum_pnl = pnl.cumsum()

        day_pnl = pnl.groupby(by=pnl.index.to_series().apply(lambda m: m.date())).sum()

        # PoT
        pot = pnl.sum() / (indicator.diff().abs().sum()) * 1e4

        # Sharpe
        sharpe = day_pnl.mean() / day_pnl.std() * 16

        # year PnL
        pnl_y = day_pnl.mean() * 250

        if st_plot:
            if ax is None:
                plt.figure(figsize=(15, 6))
                ax = plt.gca()
            sum_pnl.plot(ax=ax, label='PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                         .format(pnl_y, sharpe, pot))
            (y.cumsum() * 100.0).dropna().plot(ax=ax,
                                               label='Asset Cumulative Return', ls=':', c='k',
                                               alpha=0.5)
            ax.set_title('Simple Time-Series Test')
            ax.set_ylabel('PnL')
            ax.set_xlabel('Date')
            ax.legend()

        if bt_plot:
            get_pnl_plot(indicator, ret, **kwargs)

        if data_return:
            return {'pnl': sum_pnl, 'delta': indicator}

        return

    def simple_pnl_cs(self, x, y, dn=False, ax=None, horizon=3, intra=False, **kwargs):
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if not (isinstance(x, pd.DataFrame)) or not (isinstance(y, pd.DataFrame)):
            print('x and y must be pd.DataFrame')
            return

        if not (isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex)) \
                or not (isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex)):
            print('index of x and y must be pd.core.indexes.datetimes.DatetimeIndex')
            return

        factor, ret = _preprocessing_cs(x, y)

        # corr = _corr2(factor.stack(dropna=False), ret.stack(dropna=False))
        # factor = corr * factor

        def factor_normalize(factor_og):
            factor = factor_og.copy()

            #
            factor[np.isinf(factor)] = np.nan
            factor = factor.apply(lambda m: (m - m.mean()) / m.std(), axis=1)
            factor[factor > 4.0] = 4.0
            factor[factor < -4.0] = -4.0
            factor = factor.apply(lambda m: (m - m.mean()) / m.std(), axis=1)

            return factor

        def factor_2_delta(factor_og):
            factor = factor_og.copy()
            if dn:
                factor = factor_normalize(factor)
                #
                delta = factor.apply(lambda m: m / m.abs().sum(), axis=1) * 100.0
            else:
                # div factor std
                # delta = factor.apply(lambda m: m / m.std(), axis=0).multiply(100 / factor.count(axis=1), axis=0)
                # div factor*ret std
                # delta = factor.div((factor * ret).std(), axis=1)
                delta = factor/(factor * ret).groupby(factor.index.date).sum().sum(axis=1).std()

            return delta

        if ax is None:
            fig = plt.figure(figsize=(15, 6))
            ax = fig.add_subplot(111)

        delta = factor_2_delta(factor)
        delta_bench = pd.DataFrame(100.0, index=delta.index, columns=delta.columns).div(delta.count(axis=1), axis=0)
        delta_bench[np.isinf(delta_bench)] = np.nan
        delta_bench[np.isnan(delta)] = np.nan

        #
        pnl_bench = (delta_bench * ret).sum(axis=1)
        day_pnl_bench = pnl_bench.groupby(by=pnl_bench.index.date).sum()
        sharpe = round(day_pnl_bench.mean() / day_pnl_bench.std() * 16, 2)
        pnl_y = round(day_pnl_bench.mean() * 251, 2)
        day_pnl_bench.cumsum().plot(ax=ax, label='Asset Cumulative Return: PnL={:.2f} '
                                                 'Sharpe={:.2f}'.format(pnl_y, sharpe),
                                    ls=':', c='k', alpha=0.5)

        #
        for i in range(horizon):
            pnl = (delta * ret.shift(-i)).sum(axis=1)
            day_pnl = pnl.groupby(by=pnl.index.date).sum()

            # report
            sharpe = round(day_pnl.mean() / day_pnl.std() * 16, 2)

            pnl_y = round(day_pnl.mean() * 251, 2)

            pot = round(pnl.sum() / (delta.diff().abs().sum().sum()) * 1e4, 2)

            #
            day_pnl.cumsum().plot(ax=ax,
                                  label=f'Lag{i}: ' + 'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'.format(pnl_y, sharpe, pot))

        if intra:
            intra_delta = delta.copy()

            def _temp_apply(x):
                temp_df = x.copy()
                temp_df.iloc[-1][~ np.isnan(x.iloc[-1])] = 0.0
                return temp_df

            intra_delta = intra_delta.groupby(by=intra_delta.index.date).apply(_temp_apply)
            pnl = (intra_delta * ret).sum(axis=1)
            day_pnl = pnl.groupby(by=pnl.index.date).sum()

            # report
            sharpe = round(day_pnl.mean() / day_pnl.std() * 16, 2)

            pnl_y = round(day_pnl.mean() * 251, 2)

            pot = round(pnl.sum() / (intra_delta.diff().abs().sum().sum()) * 1e4, 2)

            #
            day_pnl.cumsum().plot(ax=ax,
                                  label=f'Intra: ' + 'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'.format(pnl_y, sharpe, pot))

        if 'title' in kwargs:
            title = kwargs.pop('title')
            ax.set_title(f'Simple PnL (info: {str(title)})')
        else:
            ax.set_title('Simple PnL')

        ax.set_ylabel('PnL')
        ax.set_xlabel('Date')
        ax.legend()

        return

    def plot_quadrant(self, factor_1, factor_2, ret):
        """
        按factor_1, factor_2符号分四象限所对应时期ret表现
        """
        fig = plt.figure(figsize=(20, 16))
        plt.title('Quandrant')
        plt.xlabel(factor_1.name)
        plt.ylabel(factor_2.name)
        plt.xticks([])
        plt.yticks([])
        #
        ax = fig.add_subplot(222)
        ret[(factor_1 >= 0) & (factor_2 >= 0)].fillna(0).cumsum().plot(ax=ax)
        ax.set_title((1, 1))
        ax.set_xlabel('')
        #
        ax = fig.add_subplot(224)
        ret[(factor_1 >= 0) & (factor_2 < 0)].fillna(0).cumsum().plot(ax=ax)
        ax.set_title((1, -1))
        ax.set_xlabel('')
        #
        ax = fig.add_subplot(223)
        ret[(factor_1 < 0) & (factor_2 < 0)].fillna(0).cumsum().plot(ax=ax)
        ax.set_title((-1, -1))
        ax.set_xlabel('')
        #
        ax = fig.add_subplot(221)
        ret[(factor_1 < 0) & (factor_2 >= 0)].fillna(0).cumsum().plot(ax=ax)
        ax.set_title((-1, 1))
        ax.set_xlabel('')
        return

    def plot_cdf(self, x, y,
                 ax=None, ax_twinx=None, with_legend=True,
                 normalize_y=False):
        """
        x: np.nddary or pd.Series
        y: np.nddary or pd.Series
        """
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()

        if isinstance(x, pd.Series) and x.name is not None:
            x_name = x.name
        else:
            x_name = 'x'

        if isinstance(y, pd.Series) and y.name is not None:
            y_name = y.name
        else:
            y_name = 'y'

        corr = _corr2(x, y)
        x_, y_ = _preprocessing(x, y)

        sorted_idx = np.argsort(x_)
        cdf_line = np.cumsum(y_.iloc[sorted_idx] - np.nanmean(y_))

        peak_idx = np.argmin(cdf_line) if corr > 0 else np.argmax(cdf_line)
        peak = x_[sorted_idx[peak_idx]]

        if normalize_y:
            y_abs_sum = np.nansum(np.abs(y_))
            cdf_line = cdf_line / y_abs_sum

        x_line = np.arange(0, len(x_))
        ax.plot(x_line, cdf_line, label=F'CDF - {y_name}')
        ax.set_xlabel('Rank')

        if ax_twinx:
            ax_ = ax_twinx
        else:
            ax_ = ax.twinx()

        ax_.plot(x_line, x_.iloc[sorted_idx], ls='--', color='orange', label=x_name)
        ax_.axhline(0, ls='--', lw=1, color='r', alpha=0.3)
        if with_legend:
            ax_.legend(loc='lower right')
        ax_.set_ylabel('Value')

        ax.text(x_line[peak_idx], cdf_line[peak_idx], 'factor val: {:.3f}'.format(peak))

        return dict(ax=ax, ax_twinx=ax_)

    def plot_markout_quantile(self, x, Y,
                              ax=None, x_regex: str = None,
                              extra_quantile=0, is_plot=True,
                              islegend=False, with_marker=False):

        x = x.copy()
        pctl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if (extra_quantile > 0) & (extra_quantile < 0.1):
            pctl = [extra_quantile] + pctl + [1 - extra_quantile]

        qtl = x.quantile(pctl).drop_duplicates()
        pctl = qtl.index
        x_q = list(sorted(qtl))

        range_list = [-np.inf] + x_q + [np.inf]
        range_list_name = [F'(-inf, {pctl[0]}]']
        for i in range(0, len(pctl) - 1):
            range_list_name.append(F'({pctl[i]}, {pctl[i + 1]}]')
        range_list_name.append(F'({pctl[-1]}, +inf)')

        if len(set(range_list)) == len(range_list):
            binned_x = pd.cut(x, range_list, labels=range_list_name)
            Y_G = Y.groupby(by=binned_x.values)
            y_mean = Y_G.mean().T * 10000
            y_mean.columns = range_list_name
        else:
            X = x.values
            y_mean = {}
            for i in range(len(range_list) - 1):
                y_mean[range_list_name[i]] = Y.loc[(range_list[i] < X) & (X <= range_list[i + 1])].mean() * 10000
            y_mean = pd.DataFrame(y_mean)

        if x_regex is not None:
            import re
            r = re.compile("nb_bars=(.*)\)")
            y_mean.index = list(map(lambda x: int(r.search(x).group(1)), y_mean.index))

        try:
            y_mean.index = y_mean.index.astype(int)
        except:
            pass

        if is_plot:
            if ax is None:
                plt.figure(figsize=(10, 10))
                ax = plt.gca()
            self.plot_y_mean(y_mean, ax, islegend=islegend, with_marker=with_marker)

        return y_mean

    def plot_y_mean(self, y_mean, ax,
                    islegend=False, with_marker=False):
        colors = plt.cm.RdBu(np.linspace(0, 1, len(y_mean.columns)))
        marker_args = dict()
        if with_marker:
            marker_args = dict(marker='.')
        if islegend:
            y_mean.plot(ax=ax, color=colors, **marker_args)
        else:
            y_mean.plot(ax=ax, color=colors, legend=None, **marker_args)

        ax.set_xlabel('Time')
        ax.set_ylabel('BPS')
        return

    def plot_markout_value(self, x, Y, values, ax=None):

        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
        range_list_name = [str(i) for i in values]
        y_mean = {}
        for i in range(len(values)):
            y_mean[range_list_name[i]] = Y.iloc[x.values == values[i]].mean() * 10000
        y_mean = pd.DataFrame(y_mean)
        y_mean.plot(ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('BPS')
        return

    def plot_markout_value_and_count(self, data, ret_all,
                                     list_value=None, ax=None):
        if list_value is None:
            self.plot_markout_value(data, ret_all.loc[data.index], list(sorted(data.unique())), ax=ax)
            plt.title(data.value_counts().sort_index(
                ascending=False).to_string().replace('\n', '; ').replace('    ', ':'))
        else:
            self.plot_markout_value(data, ret_all.loc[data.index], list_value, ax=ax)
            plt.title(data.value_counts().sort_index(
                ascending=False).reindex(list_value).fillna(0.0).to_string().replace('\n', '; ').replace('    ', ':'))

    def plot_quantile_ret(self, x, y, ax=None, is_plot=True):
        x = x.copy()
        pctl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        qtl = x.quantile(pctl).drop_duplicates()
        pctl = qtl.index
        x_q = list(sorted(qtl))

        range_list = [-np.inf] + x_q + [np.inf]
        range_list_name = [F'0.0,{pctl[0]}']
        for i in range(0, len(pctl) - 1):
            range_list_name.append(F'{pctl[i]},{pctl[i + 1]}')
        range_list_name.append(F'{pctl[-1]},1.0')

        if len(set(range_list)) == len(range_list):
            binned_x = pd.cut(x, range_list, labels=range_list_name)
            Y_G = y.groupby(by=binned_x.values)
            y_mean = Y_G.mean().T * 10000
            y_mean.columns = range_list_name
        else:
            X = x.values
            y_mean = {}
            for i in range(len(range_list) - 1):
                y_mean[range_list_name[i]] = y.loc[(range_list[i] < X) & (X <= range_list[i + 1])].mean() * 10000
            y_mean = pd.DataFrame(y_mean)

        if is_plot:
            if ax is None:
                plt.figure(figsize=(10, 10))
                ax = plt.gca()

            color_list = y_mean.apply(lambda m: 'r' if m >= 0 else 'g')
            ax.bar(y_mean.index, y_mean.values, color=color_list)
            ax.set_xlabel('Quantile')
            ax.set_ylabel('BPS')

    def hist(self, x, bins=100, ax=None, describe=False):
        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt.gca()

        if describe:
            print(x.describe())

        x.hist(bins=bins, ax=ax)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title('Factor Hist')

    def ic(self, x, y, drop_zero=False):
        if drop_zero:
            temp = x[x != 0]
            return temp.corr(y.loc[temp.index], method='spearman')
        return pd.concat([x, y], axis=1).corr(method='spearman').iloc[0, 1]

    def ir(self, x, y, window=250):
        def _rolling_ic(m):
            return pd.concat([m, y.loc[m.index]],
                             axis=1).corr(method='spearman').iloc[0, 1]

        rank_ic = x.rolling(window).apply(_rolling_ic)

        return rank_ic.mean() / rank_ic.std()

    def plot_composite(self, x, y, markout_periods=20, cdf_period2=5, fig_return=False,
                       **kwargs):
        """
        x, y: prefer pd.Series
        x: factor
        y: future ret
        """
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if not (isinstance(x, pd.Series)) or not (isinstance(y, pd.Series)):
            print('x and y must be pd.Series')
            return

        if not (isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex)) \
                or not (isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex)):
            print('index of x and y must be pd.core.indexes.datetimes.DatetimeIndex')
            return

        if 'ic' in kwargs:
            if kwargs.pop('ic'):
                print('IC: ', self.ic(x, y))

        if 'ir' in kwargs:
            if kwargs.pop('ir'):
                print('IR: ', self.ir(x, y))

        x, y = _preprocessing(x, y)

        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
        else:
            figsize = (16, 24)

        fig = plt.figure(figsize=figsize)

        Y = pd.concat([y.rolling(i).sum().shift(-i + 1).rename(str(i)) \
                       for i in range(1, markout_periods + 1)], axis=1)

        ax1 = plt.subplot2grid((3, 2), (0, 0), fig=fig)
        self.plot_cdf(x, y, ax=ax1)
        ax1.set_title('CDF 1Period')

        ax2 = plt.subplot2grid((3, 2), (0, 1), fig=fig)
        self.plot_cdf(x, y.shift(1).rolling(cdf_period2).sum().shift(-cdf_period2), ax=ax2)
        ax2.set_title('CDF {}Period'.format(cdf_period2))

        ax3 = plt.subplot2grid((3, 2), (1, 0), fig=fig)
        self.plot_markout_quantile(x, Y, ax=ax3, islegend=True)
        ax3.set_title('Markout')
        ax3.legend()

        ax4 = plt.subplot2grid((3, 2), (1, 1), fig=fig)
        self.plot_quantile_ret(x, y, ax=ax4)
        ax4.set_title('Quantile Ret Mean')

        if 'sign' in kwargs:
            sign = kwargs.pop('sign')
        else:
            sign = True

        if 'val' in kwargs:
            val = kwargs.pop('val')
        else:
            val = True

        if 'peak' in kwargs:
            peak = kwargs.pop('peak')
        else:
            peak = True

        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
        if 'title' in kwargs:
            title = kwargs.pop('title')
        else:
            title = ''
        self.simple_pnl(x, y, ax=ax5, st_plot=True, bt_plot=False,
                        sign=sign, val=val, peak=peak, title=title)
        plt.show()

        if fig_return:
            return fig

        return

    def plot_composite_cs(self, x, y,
                          markout_periods=20, cdf_period2=5, fig_return=False, horizon=3, intra=False,
                          **kwargs):
        """"""
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if not (isinstance(x, pd.DataFrame)) or not (isinstance(y, pd.DataFrame)):
            print('x and y must be pd.DataFrame')
            return

        if not (isinstance(x.index, pd.core.indexes.datetimes.DatetimeIndex)) \
                or not (isinstance(y.index, pd.core.indexes.datetimes.DatetimeIndex)):
            print('index of x and y must be pd.core.indexes.datetimes.DatetimeIndex')
            return

        x, y = _preprocessing_cs(x, y)

        if 'ic' in kwargs:
            if kwargs.pop('ic'):
                print('IC: ', self.ic(x.stack(), y.stack()))

        if 'figsize' in kwargs:
            figsize = kwargs.pop('figsize')
        else:
            figsize = (16, 24)

        fig = plt.figure(figsize=figsize)

        # cdf
        ax1 = plt.subplot2grid((3, 2), (0, 0), fig=fig)
        self.plot_cdf(x.stack(dropna=False), y.stack(dropna=False), ax=ax1)
        ax1.set_title('CDF 1Period')

        ax2 = plt.subplot2grid((3, 2), (0, 1), fig=fig)
        self.plot_cdf(x.stack(dropna=False),
                      y.shift(1).rolling(cdf_period2).sum().shift(-cdf_period2).stack(dropna=False), ax=ax2)
        ax2.set_title('CDF {}Period'.format(cdf_period2))

        # markout
        Y = pd.concat([y.rolling(i).sum().shift(-i + 1).stack(dropna=False).rename(str(i)) \
                       for i in range(1, markout_periods + 1)], axis=1)

        ax3 = plt.subplot2grid((3, 2), (1, 0), fig=fig)
        self.plot_markout_quantile(x.stack(dropna=False), Y, ax=ax3, islegend=True)
        ax3.set_title('Markout')
        ax3.legend()

        # quantile true ret
        ax4 = plt.subplot2grid((3, 2), (1, 1), fig=fig)
        self.plot_quantile_ret(x.stack(dropna=False), y.stack(dropna=False), ax=ax4)
        ax4.set_title('Quantile Ret Mean')

        # simple pnl cs
        ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
        if 'title' in kwargs:
            title = kwargs.pop('title')
        else:
            title = ''

        self.simple_pnl_cs(x, y, ax=ax5, horizon=horizon, title=title, intra=intra)

        if fig_return:
            return fig

        return
