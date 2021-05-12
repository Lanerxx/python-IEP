from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


datafile = '../sourseData/sno31-60(simliar).xlsx' #参数初始化
data = pd.read_excel(datafile) #读取数据

df1 = data['作者h指数'].values.reshape(-1, 1)
# df2 = data['引用文献时间间隔'].values.reshape(-1, 1)
df3 = data['rp'].values.reshape(-1, 1)
df4 = data['sjr'].values.reshape(-1, 1)
df5 = data['snip'].values.reshape(-1, 1)
print(data)
df6 = data['施引文献'].values.reshape(-1, 1)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
x_minmax1 = min_max_scaler.fit_transform(df1)
# x_minmax2 = min_max_scaler.fit_transform(df2)
x_minmax3 = min_max_scaler.fit_transform(df3)
x_minmax4 = min_max_scaler.fit_transform(df4)
x_minmax5 = min_max_scaler.fit_transform(df5)
x_minmax6 = min_max_scaler.fit_transform(df6)
np.set_printoptions(threshold=np.inf)
print(x_minmax1)
data['作者h指数'] = x_minmax1
# data['引用文献时间间隔'] = x_minmax2
data['rp'] = x_minmax3
data['sjr'] = x_minmax4
data['snip'] = x_minmax5
data['施引文献'] = x_minmax6
data.to_excel('../data/sno31-60forone.xlsx')


