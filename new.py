# %%
from calendar import SATURDAY
import datetime
from decimal import Decimal
from typing_extensions import runtime
import pandas as pd
import numpy as np
import seaborn as sns

# %% =================================================================
n = 10
df = pd.DataFrame(np.random.randint(n, size=(n, 2)), columns=list('bc'))
df.query('index<b<c')


# %%
n = 10
colors = np.random.choice(['red', 'green'], size=n)
foods = np.random.choice(['eggs', 'ham'], size=n)
index = pd.MultiIndex.from_arrays(colors, foods, names=['color', 'food'])
df = pd.DataFrame(np.random.randn(n, 2), index=index)


# %%
df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
df2 = pd.DataFrame(np.random.rand(n + 2, 3), columns=df.columns)
expr = '0.0 <= a <= c <= 0.5'

li = map(lambda frame: frame.query(expr), [df, df2])

# %%
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
                    'b': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],
                    'c': np.random.randn(7)})
df2.duplicated('a')
# %%
df3 = pd.DataFrame({'a': np.arange(6),
                    'b': np.random.randn(6)},
                   index=['a', 'a', 'b', 'c', 'b', 'a'])

df3.index.duplicated()
# %%
dflookup = pd.DataFrame(np.random.rand(20, 4), columns=['A', 'B', 'C', 'D'])
# %%

dflookup.lookup(list(range(0, 10, 2)), ['B', 'C', 'A', 'B', 'D'])
# %%
index = pd.Index(['e', 'd', 'c', 'b', 'a'])
index
# %%
index = pd.Index(list(range(5)), name='rows')
columns = pd.Index(['A', 'B', 'C'], name='cols')
df = pd.DataFrame(np.random.randn(5, 3), index=index, columns=columns)
df
# %%
ind = pd.Index([1, 2, 3])
ind.rename("apple")
ind.set_names(["apple"], inplace=True)


# %%
a = pd.Index(['c', 'b', 'a'])
b = pd.Index(['c', 'e', 'd'])


# %%
idx1 = pd.Index([1, np.nan, 3, 4])
idx1
idx1 = idx1.fillna(2)
idx1
# %%
idx2 = pd.DatetimeIndex([pd.Timestamp('2011-01-01'), pd.NaT,
                         pd.Timestamp('2011-01-03')])
idx2
idx2 = idx2.fillna(pd.Timestamp('2011-01-02'))
# %%
dfmi = pd.DataFrame([list('abcd'),
                     list('efgh'),
                     list('ijkl'),
                     list('mnop')],
                    columns=pd.MultiIndex.from_product([['one', 'two'],
                                                        ['first', 'second']]))
dfmi

# %%
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
s = pd.Series(np.random.randn(8), index=index)
s
# %%
iterable = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]
pd.MultiIndex.from_product(iterable, names=['first', 'second'])
# %%
df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)
pd.DataFrame(np.random.randn(6, 6), index=index[:6], columns=index[:6])
# %%
index.get_level_values(0)
index.get_level_values(1)
# %%
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])
frames = [df1, df2, df3]
res = pd.concat(frames)
# %%
result = pd.concat(frames, keys=['x', 'y', 'z'])

# %%
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])
res = pd.concat([df1, df4], axis=1, sort=False)
# %%
res2 = pd.concat([df1, df4], axis=1, join='inner')
res3 = pd.concat([df1, df4], axis=1).reindex(df1.index)
# %%
res4 = df1.append(df2, sort=False)
res4

# %%
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
res5 = pd.concat([df1, s1], axis=1)


# %%
s3 = pd.Series([0, 1, 2, 3], name='foo')
s4 = pd.Series([0, 1, 2, 3])
s5 = pd.Series([0, 1, 4, 5])
pd.concat([s3, s4, s5], axis=1, keys=['red', 'blue', 'yellow'])
# %%
s2 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])
result = df1.append(s2, ignore_index=True)
# %%----------------------------------------------------------------
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

res = pd.merge(left, right, how='inner', on='key')
res


# %%
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
res2 = pd.merge(left, right, how='inner', on=['key1', 'key2'])
res2
# %%
res3 = pd.merge(left, right, how='left', on=['key1', 'key2'])
res3

# %%
res4 = pd.merge(left, right, how='right', on=['key1', 'key2'])
res4
# %%
res5 = pd.merge(left, right, how='outer', on=['key1', 'key2'])
res5
# %%
left = pd.DataFrame({'A': [1, 2], 'B': [2, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 2, 2]})
res6 = pd.merge(left, right, on='B', how='outer')

# %%
left = pd.DataFrame({'A': [1, 2], 'B': [1, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 2, 2]})
res7 = pd.merge(left, right, on='B', how='outer', validate='one_to_many')

# %%
df1 = pd.DataFrame({'col1': [0, 1], 'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col1': [1, 2, 2], 'col_right': [2, 2, 2]})
pd.merge(df1, df2, on='col1', how='outer', indicator=True)

# %%----------------------------------------------------------------
left = pd.DataFrame({'key': [1], 'v1': [10]})
right = pd.DataFrame({'key': [1, 2], 'v1': [20, 30]})
pd.merge(left, right, on='key', how='outer')

# %%
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
result = left.join(right, how='outer')

# %%
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})


right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                     index=['K0', 'K1'])
result = left.join(right, on='key')

# %%
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=pd.Index(['K0', 'K1', 'K2'], name='key'))

index = pd.MultiIndex.from_tuples([('K0', 'Y0'), ('K1', 'Y1'),
                                   ('K2', 'Y2'), ('K2', 'Y3')],
                                  names=['key', 'Y'])

right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']},
                     index=index)

result = left.join(right, how='inner')
# %%
leftindex = pd.MultiIndex.from_product([list('abc'), list('xy'), [1, 2]],
                                       names=['abc', 'xy', 'num'])

left = pd.DataFrame({'v1': range(12)}, index=leftindex)
rightindex = pd.MultiIndex.from_product([list('abc'), list('xy')],
                                        names=['abc', 'xy'])
right = pd.DataFrame({'v2': [100 * i for i in range(1, 7)]}, index=rightindex)
right
left.join(right, on=['abc', 'xy'], how='inner')

# %%
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                    'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
# %%
df2 = df.stack()
# %%
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})
df
# %%
grouped = df.groupby('A')
grouped = df.groupby('B')


# %%
df2 = df.set_index(['A', 'B'])
# %%
lst = [1, 2, 3, 1, 2, 3]
s = pd.Series([1, 2, 3, 10, 20, 30], lst)
grouped = s.groupby(level=0)
# %%
df3 = pd.DataFrame({'X': ['A', 'B', 'A', 'B'], 'Y': [1, 4, 3, 2]})

# %%
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3],
                   'B': np.arange(8)},
                  index=index)
df.groupby([pd.Grouper(level=1), 'A']).sum()
# %%
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
s = pd.Series(np.random.randn(8), index=index)
s
# %%
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
          ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]

index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
df = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3],
                   'B': np.arange(8)},
                  index=index)
df
# %%
df.groupby([pd.Grouper(level=1), 'A']).sum()
# %%
df.groupby(['second', 'A']).sum()
# %%
grouped = df.groupby(['A'])

# %%
grouped = df.groupby('A')
for name, group in grouped:
    print(name)
    print(group)
# %%
grouped = df.groupby('A')
grouped.agg([np.sum, np.mean, np.std])

# %%
animals = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                        'height': [9.1, 6.0, 9.5, 34.0],
                        'weight': [7.9, 7.5, 9.9, 198.0]})
animals
animals.groupby("kind").agg(
    min_height=('height', 'min'),
    max_height=('height', 'max'),
    average_weight=('weight', np.mean),
)
# %%
grouped.agg({'C': np.sum,
             'D': lambda x: np.std(x, ddof=1)})

# %%
index = pd.date_range('10/1/1999', periods=1100)
ts = pd.Series(np.random.normal(0.5, 2, 1100), index=index)
ts = ts.rolling(window=100, min_periods=100).mean().dropna()

# %%
transformed = (ts.groupby(lambda x: x.year)
               .transform(lambda x: (x - x.mean()) / x.std()))
grouped = ts.groupby(lambda x: x.year)

# %%
grouped.mean()
grouped.std()

# %%
grouped.transform(lambda x: (x - x.mean()) / x.std())
# %%
test_df = pd.DataFrame(
    {'type': ['A', 'A', 'B', 'B', 'C', 'C'], "count": range(6), "count2": range(6)})
# %%
countries = np.array(['US', 'UK', 'GR', 'JP'])
key = countries[np.random.randint(0, 4, 1000)]
grouped = data_df.groupby(key)
# %%----------------------------------------------------------------
df_re = pd.DataFrame({'A': [1] * 10 + [5] * 10,
                      'B': np.arange(20)})
df_re
# %%
df_re = pd.DataFrame({'date': pd.date_range(start='2016-01-01', periods=4,
                                            freq='W'),
                      'group': [1, 1, 2, 2],
                      'val': [5, 6, 7, 8]}).set_index('date')
df_re
df_re.groupby('group').resample('1D').ffill()

# %%
sf = pd.Series([1, 1, 2, 3, 3, 3])
sf.groupby(sf).filter(lambda x: x.sum() > 2)

# %%


def tempfunc(x):
    return len(x) > 4


dff = pd.DataFrame({'A': np.arange(8), 'B': list('aabbbbcc')})
dff.groupby('B').filter(tempfunc)

# %%
sf = pd.Series([1, 1, 2, 3, 3, 3])
sf.groupby(sf).filter(lambda x: x.sum() > 2)

# %%
dff = pd.DataFrame({'A': np.arange(8), 'B': list('aabbbbcc')})
dff.groupby('B').filter(lambda x: len(x) > 2)
# %%
dff['C'] = np.arange(8)
dff.groupby('B').filter(lambda x: len(x['C']) > 2)
# %%
s = pd.Series([9, 8, 7, 5, 19, 1, 4.2, 3.3])
g = pd.Series(list('abababab'))
gb = s.groupby(g)
# %%
df = pd.DataFrame({"A": ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                  "B": ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three']})
new_df = pd.concat((df, pd.DataFrame(np.random.randn(8, 2))), axis=1)
# %%
new_df.rename(columns={0: 'C', 1: 'D'}, inplace=True)

# %%
new_df.groupby('A')['C'].apply(lambda x: x.describe())
# %%


def f(group):
    return pd.DataFrame({'original': group, 'demand': group-group.mean()})


new_df.groupby('A')['C'].apply(f)


# %%
def f(x):
    return pd.Series([x, x**2], index=['x', 'x^2'])


s = pd.Series(np.random.randn(5))
s
# %%
s.apply(f)
# %%
df_dec = pd.DataFrame(
    {'id': [1, 2, 1, 2],
     'int_column': [1, 2, 3, 4],
     'dec_column': [Decimal('0.50'), Decimal('0.15'),
                    Decimal('0.25'), Decimal('0.40')]
     }
)

df_dec.groupby('id')[['dec_column', 'int_column']].sum()
df_dec.groupby('id').agg({'int_column': 'sum', 'dec_column': 'mean'})


# %%
df = pd.DataFrame({'Branch': 'A A A A A A A B'.split(),
                   'Buyer': 'Carl Mark Carl Carl Joe Joe Joe Carl'.split(),
                   'Quantity': [1, 3, 5, 1, 8, 1, 9, 3],
                   'Date': [
    datetime.datetime(2013, 1, 1, 13, 0),
    datetime.datetime(2013, 1, 1, 13, 5),
    datetime.datetime(2013, 10, 1, 20, 0),
    datetime.datetime(2013, 10, 2, 10, 0),
    datetime.datetime(2013, 10, 1, 20, 0),
    datetime.datetime(2013, 10, 2, 10, 0),
    datetime.datetime(2013, 12, 2, 12, 0),
    datetime.datetime(2013, 12, 2, 14, 0)]
})
df

# %%
df.groupby([pd.Grouper(freq='1M', key='Date'), 'Buyer']).sum()

# %%
dti = pd.to_datetime(
    ['1/1/2018', np.datetime64('2018-01-01'), datetime.datetime(2018, 1, 1)])
dti
# %%
dti = pd.date_range('2018-01-01', periods=3, freq='H')
dti
# %%
dti = dti.tz_localize('UTC')
dti
# %%
dti.tz_convert('US/Eastern')
# %%
idx = pd.date_range('2018-01-01', periods=5, freq='H')
ts = pd.Series(range(len(idx)), index=idx)
# %%
ts.resample('2H').mean()
# %%
friday = pd.Timestamp('2018-01-05')
friday.day_name()
# %%
saturday = friday + pd.Timedelta("1 day")
saturday.day_name()
# %%
monday = friday + pd.offsets.BDay()
# %%
pd.Series(range(3), index=pd.date_range('2000', freq='D', periods=3))

# %%
pd.Series(pd.date_range('2000', freq='D', periods=3))

# %%
pd.Series(pd.period_range('1/1/2011', freq='M', periods=3))
# %%
pd.Timestamp(pd.NaT)
# %%
pd.Timestamp('2018-01-01')
# %%
pd.Period("2011-01")
pd.Period("2012-05", freq="M")
# %%
dates = [pd.Timestamp('2012-05-01'),
         pd.Timestamp('2012-05-02'), pd.Timestamp('2012-05-03')]
ts = pd.Series(np.random.randn(3), index=dates)
# %%
pd.to_datetime(pd.Series(['Jul 31, 2009', '2010-01-10', None]))

# %%
pd.to_datetime(['04-01-2012 10:00'], dayfirst=True)
# %%
pd.to_datetime('2010/11/12', format='%Y/%m/%d')

# %%
dates = [datetime.datetime(2012, 5, 1), datetime.datetime(
    2012, 5, 2), datetime.datetime(2012, 5, 3)]
index = pd.DatetimeIndex(dates)
# %%
start = datetime.datetime(2012, 5, 1)
end = datetime.datetime(2012, 5, 3)
index = pd.date_range(start, end, freq='H')
# %%
rng = pd.date_range(start, end, freq='BM')
ts = pd.Series(np.random.randn(len(rng)), index=rng)


# %%
ts = pd.Timestamp('2016-10-30 00:00:00', tz='UTC')
ts + pd.Timedelta(days=1)

# %%
two_business_day = 2 * pd.offsets.BDay()
# %%
ts = pd.Timestamp('2018-01-06 00:00:00')
ts.day_name()

# %%
