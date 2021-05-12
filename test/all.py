from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


datafile = '../sourseData/a1-5选取.xls' #参数初始化
data = pd.read_excel(datafile) #读取数据
data = data['施引文献'].values.reshape(-1, 1)

f = open('../data/data.txt', 'w')


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.01, 0.99))
x_minmax = min_max_scaler.fit_transform(data)
np.set_printoptions(threshold=np.inf)
print(x_minmax)
f.write(str(x_minmax))