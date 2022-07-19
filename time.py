# %%

import datetime
import pandas as pd
import numpy as np

# %%
ts = pd.Timestamp('2018-01-06 00:00:00')
# %%
ts.day_name()
offset = pd.offsets.BusinessHour(start='9:00', end='5:00')
offset.rollforward(ts)
ts + offset

# %%
start = datetime.datetime(2012, 5, 1)
end = datetime.datetime(2012, 5, 3)
index = pd.date_range(start, end, freq='H')

# %%
ts = pd.Series(index=index, data=np.random.randn(len(index)))
ts[:5]
# %%
ts.shift(1)
# %%
ts.shift(5, freq=pd.offsets.BDay())
ts.shift(5, freq='BM')
# %%
rng = pd.date_range('2012/1/1', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()
# %%
ts[:2].resample('250L').asfreq()
# %%

ts[:2].resample('250L').ffill()
ts[:2].resample('250L').ffill(limit=2)
# %%
rng = pd.date_range('2014-1-1', periods=100, freq='D')+pd.Timedelta("1s")
ts = pd.Series(range(100), index=rng)
# %%
pd.Period('2012', freq="A-DEC")

# %%
pd.Period('2012-01-01', freq="D")

# %%
p = pd.Period('2012-01-01 09:00',freq='H')
p+pd.offsets.Hour(2)
# %%
p + datetime.timedelta(minutes=120)
# %%
