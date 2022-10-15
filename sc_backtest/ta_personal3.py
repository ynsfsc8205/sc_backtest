import pandas as pd
import numpy as np


####################################################
# ta3: standard technical library
####################################################
# lv0
####################################################
def last(S, A, B):
    # 从前A日到前B日一直满足S_BOOL条件, 要求A>B & A>0 & B>=0
    return pd.Series(S).rolling(A + 1).apply(lambda x: np.all(x[::-1][B:]), raw=True).astype(float)


def cross(S1, S2):
    # 判断向上金叉穿越
    return ((S1 > S2) & (S1 <= S2).shift(1)).astype(float)


def long_cross(S1, S2, window=10):
    # 两条线维持一定周期后交叉,S1在N周期内都小于S2,本周期从S1下方向上穿过S2时返回1,否则返回0
    # window=1时等同于cross(S1, S2)
    return (last(S1 <= S2, window, 1).astype(bool) & (S1 > S2)).astype(float)


def bars_last(S):
    # 上一次条件S成立到当前的周期
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):
        M[i] = 0 if M[i] else M[i - 1] + 1
    M = pd.Series(M[1:], index=S.index, name=S.name)
    return M


def bars_last_count(S):
    # 统计连续满足S条件的周期数
    # bars_last_count(CLOSE>OPEN)表示统计连续收阳的周期数
    rt = np.zeros(len(S) + 1)
    for i in range(len(S)):
        rt[i + 1] = rt[i] + 1 if S[i] else rt[i + 1]
    rt = pd.Series(rt[1:], index=S.index, name=S.name)
    return rt


def bars_since(S, N):
    # N周期内第一次S条件成立到现在的周期数, N为常量
    return pd.Series(S).rolling(N).apply(lambda x: N - 1 - np.argmax(x) if np.argmax(x) or x[0] else 0, raw=True)


def value_when(S, X):
    # 当S条件成立时,取X的当前值,否则取value_when的上个成立时的X值
    output = S.copy()
    output[:] = pd.Series(np.where(S, X, np.nan)).ffill()
    return output


def top_range(S):
    # top_range(HIGH)表示当前最高价是近多少周期内最高价的最大值
    return S.expanding(2).apply(lambda x: np.argmin(np.flipud(x[:-1] <= x[-1]).astype(float)), raw=True)


def low_range(S):
    # low_range(LOW)表示当前最低价是近多少周期内最低价的最小值
    return S.expanding(2).apply(lambda x: np.argmin(np.flipud(x[:-1] >= x[-1]).astype(float)), raw=True)
####################################################
# lv1
####################################################
