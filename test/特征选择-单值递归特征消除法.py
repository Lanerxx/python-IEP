from sklearn.svm import SVC
from sklearn.feature_selection import RFE,RFECV
import pandas as pd
import numpy as np

sum = [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
count = len(sum)
datafileBase1 = '../sourseData/select_feature_data_1v4/'
datafileBase2 = '.xlsx'
# datafile = '../sourseData/select_feature_data/4.xlsx'  # 参数初始化

index = 31
total = 60
while index < total+1:
    datafile = datafileBase1 + str(index) + datafileBase2;
    print(datafile)

    data = pd.read_excel(datafile)  # 读取数据
    col = data.columns.values.tolist()
    # print(col)
    datas = np.array(data)

    # X为特征数据 y是类别值
    cols = col[1:]
    # print(cols)
    X = np.array(data[cols])
    Y = data['是否被引']

    svc = SVC(kernel="linear",C=1)
    rfe = RFE(estimator=svc,n_features_to_select=1,step=1)
    rfe.fit(X,Y)
    ranking = rfe.ranking_
    print(ranking)
    for i in range(0,count):
        indexSum = ranking[i] - 1
        sum[indexSum] = sum[indexSum] + (18 - i)
    for i in range(0,count):
        print(sum[i])
    index = index + 1
