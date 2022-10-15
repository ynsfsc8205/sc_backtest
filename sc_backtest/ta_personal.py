import math
import pandas as pd
import numpy as np
import chinese_calendar as cc
import datetime


# 游程检验 Run-test
def run_test(x):
    ss = pd.Series(x).copy()
    # 游程个数r
    ss = ss.dropna()
    if len(ss) == 0:
        return 0
    #
    ss[ss >= 0] = 1.0
    ss[ss < 0] = -1.0
    r = 1
    _init = ss.iloc[0]
    for i in range(1, len(ss)):
        if _init == ss.iloc[i]:
            continue
        else:
            _init = ss.iloc[i]
            r += 1
    return r


#
def run_test_zscore(x):
    r = run_test(x)
    #
    n1 = len(x[x >= 0])
    n2 = len(x[x < 0])
    #
    E_r = 2 * n1 * n2 / (n1 + n2) + 1
    Sigma_r = np.sqrt(2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1)))
    #
    return (r - E_r) / Sigma_r


#
def rolling_run_test(x, window=20):
    ss = pd.Series(x)
    return ss.rolling(window).apply(lambda m: run_test(m))


# z-score
def z_score(x, window=20):
    ss = pd.Series(x)
    return (ss - ss.rolling(window).mean()) / ss.rolling(window).std()


# de_mean
def de_mean(x, window=10):
    ss = pd.Series(x)
    return ss - ss.rolling(window).mean()


# div_std
def div_std(x, window=10):
    ss = pd.Series(x)
    std = ss.rolling(window).std()
    std[std == 0] = np.nan
    return ss / std


# auto_corr
def auto_corr(x, window=10, lag=1):
    ss = pd.Series(x)
    return ss.rolling(window).apply(lambda m: m.autocorr(lag=lag))


# pair_corr
def pair_corr(x, y, window=10):
    if len(x) != len(y):
        return
    ss1 = pd.Series(x)
    ss2 = pd.Series(y)
    return ss1.rolling(window).apply(lambda m: m.corr(ss2))


#
def rolling_beta(x, y, window=10):
    return x.rolling(window).apply(lambda m: m.cov(y.loc[m.index]) / m.var())


#
def rolling_alpha(x, y, window=10):
    return y.rolling(window).mean() - x.rolling(window).mean() * rolling_beta(x, y, window=window)


#
def rolling_residual(x, y, window=10):
    _y_mean = y.rolling(window).mean()
    _x_mean = x.rolling(window).mean()
    _beta = rolling_beta(x, y, window=window)
    _alpha = _y_mean - _x_mean * _beta
    return y - (_alpha + _beta * x)


#
def rolling_percentile(x, window=10):
    ss = pd.Series(x)
    return ss.rolling(window).apply(lambda m: m[m.le(m.iloc[-1])].count() / m.count())


#
def get_beta(x, y):
    return x.cov(y) / x.var()


#
def get_alpha(x, y):
    return y.mean() - x.mean() * get_beta(x, y)


#
def get_residual(x, y):
    return y - (get_beta(x, y) * x + get_alpha(x, y))


#
def rolling_poly_param(x, window=10, deg=1, ind=1):
    if ind > deg:
        return
    return x.rolling(window).apply(lambda m: np.polyfit(np.linspace(1, window, num=window), m, deg=deg)[deg - ind])


# 计算样本内量分布
def volume_distribution(price, volume, M=5):
    range_list = np.linspace(price.min() - 0.02, price.max() + 0.02, M + 1)
    range_name = range(M)
    return volume.groupby(pd.cut(price, range_list, labels=range_name)).sum()


# 计算滚动窗口区内量分布
def rolling_volume_distribution(price, volume, N=20, M=5):
    output_list = []

    def _volume_distribution(price, volume, M):
        range_list = np.linspace(price.min() - 0.02, price.max() + 0.02, M + 1)
        range_name = range(M)
        output_list.append(volume.groupby(pd.cut(price, range_list, labels=range_name)).sum().rename(price.index[-1]))
        return np.nan

    price.rolling(N).apply(lambda x: _volume_distribution(x, volume.loc[x.index], M=M))
    return pd.concat(output_list, axis=1).T.reindex(price.index)


#
def poly_param(x, deg=1, ind=1):
    ss = x.dropna()
    if len(ss) != len(x):
        return np.nan
    return np.polyfit(np.linspace(1, len(ss), num=len(ss)), ss, deg=deg)[deg - ind]


#
def pair_poly_param(x, y, deg=1, ind=1):
    ss1 = x.dropna()
    ss2 = y.dropna()
    if len(ss1) != len(ss2):
        return np.nan
    return np.polyfit(ss1, ss2, deg=deg)[deg - ind]


#
def rsi(x, window=10):
    temp = pd.Series(x)
    U = temp.diff()
    U[U < 0] = 0
    D = temp.diff()
    D[D > 0] = 0
    D = D.abs()

    def _smma(s, w):
        return pd.Series(s).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()

    RS = _smma(U, window) / _smma(D, window)

    return -100 / (1 + RS) + 100


# average true range
def atr(high, low, close, window=10):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# cmo
def cmo(x_, window=10):
    ss = pd.Series(x_).copy()
    #
    SU = ss.rolling(window + 1).apply(lambda x: x.diff()[x.diff() > 0].sum())
    SD = ss.rolling(window + 1).apply(lambda x: -x.diff()[x.diff() < 0].sum())
    CMO = (SU - SD) / (SU + SD) * 100.0
    return CMO


# adx
def adx(high, low, close, window=10):
    #
    dm_plus = close.copy()
    dm_plus[:] = 0.0
    dm_plus[(high.diff() > -low.diff()) & (high.diff() > 0)] = high.diff()
    #
    dm_minus = close.copy()
    dm_minus[:] = 0.0
    dm_minus[(high.diff() < -low.diff()) & (low.diff() < 0)] = -low.diff()
    #
    tr = pd.concat([high - low, low - close.shift(), high - close.shift()], axis=1).abs().max(axis=1)

    def _smma(s, w):
        return pd.Series(s).ewm(alpha=1 / w, adjust=False, min_periods=w).mean()

    di_plus = _smma(dm_plus, window) / _smma(tr, window) * 100.0
    di_minus = _smma(dm_minus, window) / _smma(tr, window) * 100.0

    dx = (di_plus - di_minus).abs() / (di_plus + di_minus) * 100.0
    ADX = _smma(dx, window)
    return ADX


# ts rank
def ts_rank(x, window=10):
    return pd.Series(x).rolling(window).apply(lambda m: m.rank().iloc[-1])


# Simple ma
def sma(x, window=10):
    return pd.Series(x).rolling(window).mean()


# Linear weighted ma
def wma(x, window=10):
    coef = 2.0 / (window * (window + 1.0))  # sum of weights in a fancy way n(n+1)/2
    weights = list(float(i) for i in range(1, window + 1))
    return pd.Series(x).rolling(window).apply(lambda x: np.sum(weights * x) * coef, raw=True)


# Exponential ma
def ema(x, window=10):
    return pd.Series(x).ewm(span=window, adjust=False).mean()


# macd
def macd(x, window1=12, window2=26, window3=9):
    ss = pd.Series(x)
    dma_line = ema(ss, window=window1) - ema(ss, window=window2)
    return dma_line - ema(dma_line, window=window3)


# N is replaced in the code by window
def mma(x, window):
    return x.rolling(window).median()


def gma(x, window):
    return x.rolling(window).apply(lambda u: (np.prod(u)) ** (1 / window))


def qma(x, window):
    return x.rolling(window).apply(lambda u: (np.sum(u ** 2) / window) ** 0.5)


def hama(x, window):
    return x.rolling(window).apply(lambda u: (window / np.sum(1 / u)))


def trima(serie, window):
    return sma(sma(serie, window), window)


def swma(serie, window):
    weights = np.arange(1, window + 1) * (np.pi / 6)
    tmp = pd.Series(weights).apply(lambda x: round(math.sin(x)))
    #     return serie.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum())
    return serie.rolling(window).apply(lambda x: np.dot(x, tmp) / tmp.sum())


def zlema(serie, window):
    lag = (window - 1) / 2
    p = serie + serie.diff(lag)
    return ema(p, window)


def hma(serie, window):
    half_window = int(window / 2)
    sqrt_window = int(math.sqrt(window))
    wma_f = wma(serie, window=half_window)
    wma_s = wma(serie, window=window)
    return wma(2 * wma_f - wma_s, window=sqrt_window)


def ehma(serie, window):
    half_window = int(window / 2)
    sqrt_window = int(math.sqrt(window))
    ema_f = ema(serie, window=half_window)
    ema_s = ema(serie, window=window)
    return ema(2 * ema_f - ema_s, window=sqrt_window)


# First implementation
def gd(serie, window):
    ema1 = ema(serie, window)
    ema2 = ema(ema1, window)
    v = 0.618
    return (1 + v) * ema1 - ema2 * v


def tima(serie, window):
    gd1 = gd(serie, window)
    gd2 = gd(gd1, window)
    gd3 = gd(gd2, window)
    return gd3


# another implementation
def tima_2(serie, window):
    ema1 = ema(serie, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    ema4 = ema(ema3, window)
    ema5 = ema(ema4, window)
    ema6 = ema(ema5, window)
    a = 0.618
    t3 = -(a ** 3) * ema6 + 3 * (a ** 2 + a ** 3) * ema5 + (-6 * (a ** 2) - 3 * a - 3 * (a ** 3)) * ema4 + (
            1 + 3 * a + a ** 3 + 3 * (a ** 2)) * ema3
    return t3


# The Kaufman Efficiency indicator
def er(serie, window):
    x = serie.diff(window).abs()
    y = serie.diff().abs().rolling(window).sum()
    return x / y


def kama(serie, window, fast_win=2, slow_win=30):
    er_ = er(serie, window)
    fast_alpha = 2 / (fast_win + 1)  # = 0,6667
    slow_alpha = 2 / (slow_win + 1)  # = 0,0645
    sc = pd.Series((er_ * (fast_alpha - slow_alpha) + slow_alpha) ** 2)  ## smoothing constant
    sma_ = sma(serie, window)  ## first KAMA is SMA
    kama_ = []
    for s, ma, price in zip(
            sc.iteritems(), sma_.shift().iteritems(), serie.iteritems()):
        try:
            kama_.append(kama_[-1] + s[1] * (price[1] - kama_[-1]))
        except (IndexError, TypeError):
            if pd.notnull(ma[1]):
                kama_.append(ma[1] + s[1] * (price[1] - ma[1]))
            else:
                kama_.append(None)
    return pd.Series(kama_, index=sma_.index)


def bma(serie, window):
    serie = serie.dropna()
    beta = 2.415 * (1 - np.cos((2 / window) * np.pi))
    alpha = -beta + math.sqrt(beta ** 2 + 2 * beta)
    c_0 = (alpha ** 2) / 4
    a1 = 2 * (1 - alpha)
    a2 = -(1 - alpha) ** 2

    bma_ = [serie[0], serie[1], serie[2]]
    for i in range(3, len(serie)):
        bma_.append((serie[i] + 2 * serie[i - 1] + serie[i - 2]) * c_0 + a1 * bma_[-1] + a2 * bma_[-2])

    return pd.Series(bma_, index=serie.index)


def vidya(serie, window):
    serie = serie.tail(3 * window)
    win_f = window
    win_s = 2 * win_f
    vidya_tmp = [serie.iloc[win_s]]
    for i in range(win_s + 1, len(serie)):
        s = 0.2
        if serie.iloc[i - win_s:i].std() == 0:
            return pd.Series(np.nan)
        k = serie.iloc[i - win_f:i].std() / serie.iloc[i - win_s:i].std()
        alpha = k * s
        vidya_tmp.append(alpha * serie.iloc[i] + (1 - alpha) * vidya_tmp[-1])
    return pd.Series(vidya_tmp, index=serie[win_s:].index)


def is_trading_day(timestamp):
    if isinstance(timestamp, datetime.datetime) or \
            isinstance(timestamp, datetime.date):
        return cc.is_workday(timestamp) and timestamp.weekday() <= 4
    else:
        print('input type should be datetime.datetime or datetime.date')
        return False


def not_trading_day(timestamp):
    if isinstance(timestamp, datetime.datetime) or \
            isinstance(timestamp, datetime.date):
        return not (cc.is_workday(timestamp) and timestamp.weekday() <= 4)
    else:
        print('input type should be datetime.datetime or datetime.date')
        return False


def next_trading_day(timestamp):
    if isinstance(timestamp, datetime.datetime) or \
            isinstance(timestamp, datetime.date):
        dt = timestamp
        while True:
            dt += datetime.timedelta(days=1)
            if is_trading_day(dt):
                break
        return dt
    else:
        print('input type should be datetime.datetime or datetime.date')
        return None


def get_settle_day(s):
    """
    :param s: a str: 'IF1601', 'IC1701'
    :return: datetime.datetime: settle day
    """

    def str_clip(s):
        for i in range(len(s)):
            if s[i].isdecimal():
                break
        return s[i:] if i >= 0 else ''

    sd = datetime.datetime.strptime(f'20{str_clip(s)}01', '%Y%m%d')
    wd = sd.weekday()
    if wd <= 4:
        settle_day = datetime.datetime(sd.year, sd.month, 5 - wd + 14, 15, 0)
    else:
        settle_day = datetime.datetime(sd.year, sd.month, 7 - wd + 19, 15, 0)
    try:
        if cc.is_holiday(settle_day):
            i = 3
            while True:
                temp = settle_day + datetime.timedelta(days=i)
                if cc.is_workday(temp):
                    settle_day = temp
                    break
                else:
                    i += 1
    except:
        pass

    return settle_day


def get_rank_delta(factor):
    delta = pd.DataFrame(factor)
    delta = delta.rank(axis=1)
    delta = delta.sub(delta.mean(axis=1), axis=0)
    delta = delta.div(delta.abs().sum(axis=1), axis=0) * 100
    return delta


def get_normalize_delta(factor, window=60 * 48):
    data = pd.DataFrame(factor)

    data[np.isinf(data.fillna(0.0))] = np.nan

    data = data / data.rolling(window).std()

    data[data > 3] = 3

    data[data < -3] = -3

    delta = data * 100
    return delta


# factor to indicator
##################################################
# aberration策略
def factor_to_aberration(factor, window=20, std_param=1.5):
    ss = pd.Series(factor).copy()
    indicator = pd.Series(factor).copy()
    indicator[:] = np.nan
    #
    mid = ss.rolling(window).mean()
    std = ss.rolling(window).std()
    up = mid + std_param * std
    down = mid - std_param * std
    #
    indicator[(np.sign(ss - mid) != np.sign((ss - mid).shift()))] = 0.0
    indicator[ss >= up] = 1.0
    indicator[ss <= down] = -1.0

    return indicator.ffill()


# 海归策略
def factor_to_turtle(factor, window=5):
    ss = pd.Series(factor).copy()
    indicator = pd.Series(factor).copy()
    indicator[:] = 0.0
    #
    indicator[ss == ss.rolling(window).max()] = 1.0
    indicator[ss == ss.rolling(window).min()] = -1.0
    return indicator


# 钱德动量指标CMO策略
def factor_to_cmo(factor, window=10):
    ss = pd.Series(factor).copy()
    indicator = pd.Series(factor).copy()
    indicator[:] = np.nan
    #
    CMO = cmo(ss, window=window)
    indicator[CMO >= 0] = 1.0
    indicator[CMO < 0] = -1.0
    return indicator


# 双均线策略
def factor_to_2ma(factor, window1=5, window2=20):
    ss = pd.Series(factor).copy()
    indicator = pd.Series(factor).copy()
    indicator[:] = np.nan
    #
    ma1 = ss.rolling(window1).mean()
    ma2 = ss.rolling(window2).mean()
    indicator[ma1 >= ma2] = 1.0
    indicator[ma1 < ma2] = -1.0
    return indicator


# rsi1 超卖超买
def factor_to_rsi1(factor, window=10, upper=70, lower=30):
    ss = pd.Series(factor).copy()
    indicator = pd.Series(factor).copy()
    indicator[:] = np.nan
    #
    RSI = rsi(ss, window=window)
    #
    indicator[(RSI > 50)] = 1.0
    indicator[(RSI < 50)] = -1.0
    indicator[(RSI < upper) & (RSI.shift() >= upper)] = -1.0
    indicator[(RSI > lower) & (RSI.shift() <= lower)] = 1.0
    indicator[(RSI > 50)] = 1.0
    indicator[(RSI > lower) & (RSI.shift() <= lower)] = 1.0
    return indicator.ffill()