# #!/usr/bin/env python
# #-*- coding:utf-8 -*-
# # author:wanglubao
# # datetime:2019/9/22 14:31
# # software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#加载数据
# def loadDataSet(fileName):
#     data = np.loadtxt(fileName,delimiter='\t')
#     #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
#     return data
#     #data =  np.loadtxt(fileName)
#
# #欧氏距离计算
# def distEclud(x,y):
#     return np.sqrt(np.sum((x-y)**2))
# # 为给定数据集构建一个包含K个随机质心的集合
# def randCent(dataSet,k):
#     # 获取样本数与特征值
#     m,n = dataSet.shape#把数据集的行数和列数赋值给m,n
#     # 初始化质心,创建(k,n)个以零填充的矩阵
#     centroids = np.zeros((k,n))
#     # 循环遍历特征值
#     for i in range(k):
#         index = int(np.random.uniform(0,m))
#         # 计算每一列的质心,并将值赋给centroids
#         centroids[i,:] = dataSet[index,:]
#         # 返回质心
#     return centroids
#
#
# # k均值聚类
# def KMeans(dataSet,k):
#     m = np.shape(dataSet)[0]
#     # 初始化一个矩阵来存储每个点的簇分配结果
#     # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
#     clusterAssment = np.mat(np.zeros((m,2)))
#     clusterChange = True
#
#     # 创建质心,随机K个质心
#     centroids = randCent(dataSet,k)
#     # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
#     while clusterChange:
#         clusterChange = False
#
#         #遍历所有样本（行数）
#         for i in range(m):
#             minDist = 100000.0
#             minIndex = -1
#             # 遍历所有数据找到距离每个点最近的质心,
#             # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
#             for j in range(k):
#                 # 计算数据点到质心的距离
#                 # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
#                 distance = distEclud(centroids[j,:],dataSet[i,:])
#                 # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
#                 if distance < minDist:
#                     minDist = distance
#                     minIndex = j
#             # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
#             if clusterAssment[i,0] != minIndex:
#                 clusterChange = True
#                 # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
#                 clusterAssment[i,:] = minIndex,minDist
#         # 遍历所有质心并更新它们的取值
#         for j in range(k):
#             # 通过数据过滤来获得给定簇的所有点
#             pointsInCluster = dataSet[np.nonzero(clusterAssment[:,0].A == j)[0]]
#             # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
#             centroids[j,:] = np.mean(pointsInCluster,axis=0)
#     print("Congratulation,cluster complete!")
#     # 返回所有的类质心与点分配结果
#     return centroids,clusterAssment
#
# def showCluster(dataSet,k,centroids,clusterAssment):
#     m,n = dataSet.shape
#     #if n != 2:
#        # print("数据不是二维的")
#         #return 1
#
#     mark = ['or','ob','og','ok','^r','+r','sr','dr','<r','pr']
#     if k > len(mark):
#         print("k值太大了")
#         return 1
#     #绘制所有样本
#     for i in range(m):
#         markIndex = int(clusterAssment[i,0])
#         plt.plot(dataSet[i,0],dataSet[i,1],mark[markIndex])
#
#     mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
#     #绘制质心
#     for i in range(k):
#         plt.plot(centroids[i,0],centroids[i,1],mark[i])
#
#     plt.show()
# dataSet = loadDataSet('../sourseData/cc.txt')
# k = 3
# centroids,clusterAssment = KMeans(dataSet,k)
# showCluster(dataSet,k,centroids,clusterAssment)

# --------------=============--------------=============--------------=============--------------=============
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
def loadDataSet(fileName):
    data = np.loadtxt(fileName,delimiter='\t')
    #data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
    return data
    #data =  np.loadtxt(fileName)
dataSet = loadDataSet('../sourseData/cc.txt')
# X = [[0.0888, 0.5885],
#      [0.1399, 0.8291],
#      [0.0747, 0.4974],
#      [0.0983, 0.5772],
#      [0.1276, 0.5703],
#      [0.1671, 0.5835],
#      [0.1906, 0.5276],
#      [0.1061, 0.5523],
#      [0.2446, 0.4007],
#      [0.1670, 0.4770],
#      [0.2485, 0.4313],
#      [0.1227, 0.4909],
#      [0.1240, 0.5668],
#      [0.1461, 0.5113],
#      [0.2315, 0.3788],
#      [0.0494, 0.5590],
#      [0.1107, 0.4799],
#      [0.2521, 0.5735],
#      [0.1007, 0.6318],
#      [0.1067, 0.4326],
#      [0.1956, 0.4280]
#      ]
X= dataSet
# print(X)

# Kmeans聚类
clf = KMeans(n_clusters=5)
y_pred = clf.fit_predict(X)
print(clf)
print(y_pred)

import numpy as np
import matplotlib.pyplot as plt

x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)
# z = [n[2] for n in X]
# print(z)

# 可视化操作
plt.scatter(x, y, c=y_pred, marker='x')
plt.title("Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend(["Rank"])
plt.show()
