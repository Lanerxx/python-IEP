from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit

# 导入数据集中的数据（每项有18个特征数据值，1个目标类别值）
datafile = '../sourseData/1-100-1v4-特征选择递归特征消除法-微观.xlsx' #参数初始化
data = pd.read_excel(datafile) #读取数据
col = data.columns.values.tolist()
datas=np.array(data)
# X为特征数据 y是类别值
cols =col[2:-1]
#print(cols)
X = np.array(data[cols])
Y = data['是否被引']
# 交叉分类
# train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25)

# KFold方法  k折交叉验证 0.8864142538975501
# kf = KFold(n_splits=2)
# for train_index, test_index in kf.split(X):
#     print('train_index', train_index, 'test_index', test_index)
#     train_X, train_Y = X[train_index], Y[train_index]
#     test_X, test_Y = X[test_index], Y[test_index]

# RepeatedKFold  p次k折交叉验证 0.973214286
kf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
for train_index, test_index in kf.split(X):
    print('train_index', train_index, 'test_index', test_index)
    train_X, train_Y = X[train_index], Y[train_index]
    test_X, test_Y = X[test_index], Y[test_index]

# LeaveOneOut 留一法 跑不完
# loo = LeaveOneOut()
# for train_index, test_index in loo.split(X):
#     print('train_index', train_index, 'test_index', test_index)
#     train_X, train_Y = X[train_index], Y[train_index]
#     test_X, test_Y = X[test_index], Y[test_index]

# LeavePOut  留P法 跑不完
# lpo = LeavePOut(p=500)
# for train_index, test_index in lpo.split(X):
#     print('train_index', train_index, 'test_index', test_index)
#     train_X, train_Y = X[train_index], Y[train_index]
#     test_X, test_Y = X[test_index], Y[test_index]

# ShuffleSplit  随机分配 0.9155555555555556
# ss = ShuffleSplit(n_splits=4, random_state=0, test_size=0.25)
# for train_index, test_index in ss.split(X):
#     print('train_index', train_index, 'test_index', test_index)
#     train_X, train_Y = X[train_index], Y[train_index]
#     test_X, test_Y = X[test_index], Y[test_index]

# 总数据条数 遍历显示
#icount = X.shape[0]
#for i in range(icount - 1):
#    print(X[i], "--->", Y[i])
#print(icount)

# 调用高斯朴素贝叶斯分类器
gnb = GaussianNB()


# gnb.fit(train_X, train_Y)
# # 训练完后预测
# y_predicted = gnb.predict(test_X)
# # 显示预测结果
# print("\n预测结果:\n", y_predicted)
#
# # 显示预测错误率
# print("\n总数据%d条 预测失误%d条" % (datas.data.shape[0], (test_Y!= y_predicted).sum()))
# #获取结果
# print ('The Accuracy of Naive Bayes Classifier is:', gnb.score(test_X,test_Y))
# print (classification_report(test_Y, y_predicted, target_names = ['0', '1']))
datafile11 = '../sourseData/s2018/76.XLSX'
data11 = pd.read_excel(datafile11) #读取数据
col11 = data.columns.values.tolist()
datas11 = np.array(data11)
# X为特征数据 y是类别值
cols11 =col11[2:-1]
#print(cols)
X11 = np.array(data11[cols11])
Y11 = data11['是否被引']

gnb.fit(X, Y)
# 训练完后预测
y_predicted = gnb.predict(X11)

# 显示预测结果
print("\n预测结果:\n", y_predicted)

# 显示预测错误率
print("\n总数据%d条 预测失误%d条" % (datas11.data.shape[0], (Y11!= y_predicted).sum()))
#获取结果
print ('The Accuracy of Naive Bayes Classifier is:', gnb.score(X11,Y11))
print (classification_report(Y11, y_predicted, target_names = ['0', '1']))


