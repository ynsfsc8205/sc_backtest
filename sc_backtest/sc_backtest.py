# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:14:23 2021

@author: chang.sun
"""
# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class sc_backtest():
    '''
    Index Future Backtesting Module
    '''

    def __init__(self, is_day=True, freq=1):
        self.is_day = is_day
        self.freq = freq

    def simple_pnl(self, x, y,
                   ax=None, plot=True):
        '''
        x: pd.Series
        y: pd.Series
        '''

        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        corr = pd.concat([x, y], axis=1).dropna().corr().iloc[0, 1]
        factor = x.copy()

        indicator1 = np.sign(factor) * 100 * np.sign(corr)
        label1 = 'Sign-Trade: '

        indicator2 = factor / factor.abs().mean() * 100 * np.sign(corr)
        label2 = 'Value-Trade: '

        pnl1 = y * indicator1
        sum_pnl1 = pnl1.cumsum()

        pnl2 = y * indicator2
        sum_pnl2 = pnl2.cumsum()

        if self.is_day:
            day_pnl1 = pnl1
            day_pnl2 = pnl2
        else:
            day_pnl1 = sum_pnl1[self.freq - 1::self.freq].diff()
            day_pnl2 = sum_pnl2[self.freq - 1::self.freq].diff()

        # PoT
        pot1 = pnl1.sum() / (indicator1.diff().abs().sum()) * 1e4
        pot2 = pnl2.sum() / (indicator2.diff().abs().sum()) * 1e4

        # Sharpe
        sharpe1 = day_pnl1.mean() / day_pnl1.std() * 16
        sharpe2 = day_pnl2.mean() / day_pnl2.std() * 16

        # year PnL
        pnl_y1 = day_pnl1.mean() * 250
        pnl_y2 = day_pnl2.mean() * 250

        if plot:
            if ax is None:
                ax = plt.gca()
            sum_pnl1.plot(ax=ax, label=label1 +
                                       'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                          .format(pnl_y1, sharpe1, pot1))

            sum_pnl2.plot(ax=ax, label=label2 +
                                       'PnL={:.2f} Sharpe={:.2f} PoT={:.2f}'
                          .format(pnl_y2, sharpe2, pot2))
            (y.cumsum() * 100.0).dropna().plot(ax=ax,
                                               label='Y Cumulative Return', ls=':', c='k',
                                               alpha=0.5)
            ax.set_title('Simple Time-Series Test')
            ax.set_ylabel('PnL')
            ax.set_xlabel('Date')
            ax.legend()

        return ax

    def plot_cdf(self, x, y,
                 ax=None, ax_twinx=None, with_legend=True,
                 with_label=True, normalize_y=True):
        '''
        x: np.addary or pd.Series
        y: np.addary or pd.Series
        '''
        if len(x) != len(y):
            print(F'mismatched length of x, y, len(x) == {len(x)} len(y) == {len(y)}')
            return

        if ax is None:
            ax = plt.gca()

        if isinstance(x, pd.Series) and x.name is not None:
            x_name = x.name
        else:
            x_name = 'x'

        if isinstance(y, pd.Series) and y.name is not None:
            y_name = y.name
        else:
            y_name = 'y'

        valid_x = np.isfinite(x)
        x_ = x[valid_x]
        y_ = y[valid_x]

        sorted_idx = np.argsort(x_)
        cdf_line = np.cumsum(y_.iloc[sorted_idx] - np.nanmean(y_))

        if normalize_y:
            y_abs_sum = np.nansum(np.abs(y_))
            cdf_line = cdf_line / y_abs_sum

        x_line = np.arange(0, len(x_))
        ax.plot(x_line, cdf_line, label=F'CDF - {y_name}')

        if ax_twinx:
            ax_ = ax_twinx
        else:
            ax_ = ax.twinx()

        ax_.plot(x_line, x_.iloc[sorted_idx], ls='--', color='orange', label=x_name)
        ax_.axhline(0, ls='--', lw=1, color='r', alpha=0.3)
        if with_legend:
            ax_.legend(loc='lower right')

        return dict(ax=ax, ax_twinx=ax_)

    def plot_markout_quantile(self, x, Y, ax=None, x_regex: str = None,
                              extra_quantile=0, is_plot=True,
                              islegend=False, with_marker=False):

        x = x.copy()
        pctl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if ((extra_quantile > 0) & (extra_quantile < 0.1)):
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
                ax = plt.gca()
            self.plot_y_mean(y_mean, ax, islegend=islegend, with_marker=with_marker)

        return y_mean

    def plot_y_mean(self, y_mean, ax, islegend=False, with_marker=False):
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

    def plot_markout_value(self, x, Y, values, ax=None):

        if ax is None:
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

    def hist(self, x, bins=100):
        plt.figure()
        x.hist(bins=bins)

    def plot_composite(self, x, y, markout_periods=30, fig=None, figsize=(20, 20)):
        '''
        x, y: pd.Series
        x: factor
        y: future ret
        '''
        if (not isinstance(x, pd.Series)) or (not isinstance(y, pd.Series)):
            print('x, y must be pd.Series!')
            return

        if fig is None:
            fig = plt.figure(figsize=figsize)

        Y = pd.concat([y.rolling(i).sum().shift(-i + 1).rename(str(i)) \
                       for i in range(1, markout_periods + 1)], axis=1)

        ax1 = fig.add_subplot(221)
        self.plot_cdf(x, y, ax=ax1)
        ax1.set_title('CDF 1Period')

        ax2 = fig.add_subplot(222)
        self.plot_cdf(x, y.shift(1).rolling(5).sum().shift(-5), ax=ax2)
        ax2.set_title('CDF 5Period')

        ax3 = fig.add_subplot(223)
        self.plot_markout_quantile(x, Y, ax=ax3, islegend=True)
        ax3.set_title('Markout')
        ax3.legend()

        ax4 = fig.add_subplot(224)
        self.simple_pnl(x, y, ax=ax4, plot=True)
        plt.show()

        return fig
