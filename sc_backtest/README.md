# Index Futures Simple Backtest Module (Personal Usage)

Chang Sun 
[Email](ynsfsc@126.com)

## Install and Update
```
pip install --upgrade sc-backtest
```
or (if slow)
```
pip install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple sc-backtest
```

## Simple Test
* Check for factor validity
   * Statistical:
      * CDF
      * Markout
      * Hist
      * ...
   * Time-Series:
      * Sign-Trade
      * Value-Trade
      * Threshold-Trade
      * ...	

```
# x: factors
# y: asset's future ret

import pandas as pd
import numpy as np
from sc_backtest import simpletest

data = pd.read_csv('.\factor_and_ret.csv', index_col=0, header=0)
x = data.loc[:, 'factor']
y = data.loc[:, 'ret']
st = simpletest()
st.plot_cdf(x, y)
st.plot_composite(x, y)
```

## Backtest (bt)
* Backtest
   * get_report
   * get_pnl_plot
   * round_test
   * ...

```
# x: factors
# y: asset's future ret

import pandas as pd
import numpy as np
from sc_backtest import simpletest, bt

data = pd.read_csv('.\factor_and_ret.csv', index_col=0, header=0)
x = data.loc[:, 'factor']
y = data.loc[:, 'ret']
st = simpletest()
data = st.simple_pnl(x, y, data_return=True)
report = bt.get_report(data['delta_med'], y)
bt.get_pnl_plot(data['delta_med'], y)
```

## Technical Analysis (ta)
Reference: [ta](https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html)

## Technical Analysis2 (ta2)
Variou moving average function and stat model
* SMA
* EMA
* WMA
* MMA
* QMA
* Z-Score
* ...
```
import pandas as pd
import numpy as np
from sc_backtest import ta2

wma = ta2.wma(pd.Series(np.random.rand(100)), window=5)
```


## Example
Input your factor and underlying asset's future return series with index type as DatetimeIndex and get the composite stat and time-series plots.
```
# x: factors
# y: asset's future ret

import pandas as pd
import numpy as np
from sc_backtest import simpletest, bt

data = pd.read_csv('.\factor_and_ret.csv', index_col=0, header=0)
x = data.loc[:, 'factor']
y = data.loc[:, 'ret']
st = simpletest()
st.plot_composite(x, y, markout_periods=30, cdf_period2=5)
delta = np.sign(x)*100
bt.get_pnl_plot(delta, y, alpha=True)
```
