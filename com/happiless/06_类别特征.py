import pandas as pd
from sklearn import linear_model

df = pd.DataFrame({
    'City':
        ['SF', 'SF', 'SF', 'NYC', 'NYC', 'NYC', 'Seattle', 'Seattle', 'Seattle'],
    'Rent': [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
})
df['Rent'].mean()
print(df.head())
# 独热编码
one_hot_df = pd.get_dummies(df, prefix=['City'])
print(one_hot_df.head())

# 虚拟编码
dummy_df = pd.get_dummies(df, prefix=['City'], drop_first=True)
print(dummy_df)

model = linear_model.LinearRegression() 
model.fit(one_hot_df[['City_NYC', 'City_SF', 'City_Seattle']], one_hot_df[['Rent']])
print(f'w: {model.coef_}, b: {model.intercept_}')

model.fit(dummy_df[['City_SF', 'City_Seattle']], dummy_df[['Rent']])
print(f'w: {model.coef_}, b: {model.intercept_}')

# 效果编码
effect_df = dummy_df.copy()
effect_df.loc[3:5, ['City_SF', 'City_Seattle']] = -1.
print(effect_df)

model.fit(effect_df[['City_SF', 'City_Seattle']], effect_df[['Rent']])
print(f'w: {model.coef_}, b: {model.intercept_}')


# 处理大型分类变量
# hash
def hash_features(word_list, m):
    output = [0] * m

    for word in word_list:
        index = hash(word) % m
        output[index] += 1

    return output


from sklearn.feature_extraction import FeatureHasher

m = len(df.City.unique())
handler = FeatureHasher(n_features=m, input_type='string')
f = handler.transform(df['City'])
print(f.toarray())

from sys import getsizeof

print('Our pandas Series, in bytes: ', getsizeof(df['City']))
print('Our hashed numpy array, in bytes: ', getsizeof(f))

# bin-counting
import pandas as pd
import numpy as np

# 读取前面的10k行
df = pd.read_csv('../../data/train_subset.csv')
# 有多少独立的特征
print(len(df['device_id'].unique()))


def click_counting(x, bin_column):
    clicks = pd.Series(x[x['click'] > 0][bin_column].value_counts(), name='clicks')
    no_clicks = pd.Series(x[x['click'] < 1][bin_column].value_counts(), name='no_clicks')
    # clicks = x[x['click'] > 0][bin_column].value_counts()
    # no_clicks = x[x['click'] < 1][bin_column].value_counts()
    # counts = pd.DataFrame([clicks, no_clicks], index=['click', 'no_click']).T.fillna(0)
    counts = pd.DataFrame([clicks, no_clicks]).T.fillna(0)
    counts['total'] = counts['clicks'].astype('int64') + counts['no_clicks'].astype('int64')
    return counts


def bin_counting(x):
    x['N+'] = x['clicks'].astype('int64') / x['total']
    x['N-'] = x['no_clicks'].astype('int64') / x['total']
    x['log_N+'] = x['N+'] / x['N-']

    bin_counts = x.filter(items=['N+', 'N-', 'log_N+'])
    return x, bin_counts


bin_column = 'device_id'
device_clicks = click_counting(df.filter(items=[bin_column, 'click']), bin_column)
device_all, device_bin_counts = bin_counting(device_clicks)

print(len(device_bin_counts))
print(device_all.sort_values(by='total', ascending=False).head())

print('Our pandas Series, in bytes: ',
      getsizeof(df.filter(items=['device_id', 'click'])))
print('Our bin-counting feature, in bytes: ', getsizeof(device_bin_counts))
