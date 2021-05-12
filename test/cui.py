from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# 导入数据集中的数据（每项有18个特征数据值，1个目标类别值）
datafile = '../sourseData/特征.xlsx' #参数初始化
datafile2='../sourseData/特征.xlsx'
data = pd.read_excel(datafile) #读取数据
data2=pd.read_excel(datafile2)

col = data.columns.values.tolist()
col2 = data2.columns.values.tolist()

datas=np.array(data)
datas2=np.array(data2)
# X为特征数据 y是类别值

cols =col[1:-1]
cols2 =col2[1:-1]
#print(cols)
X = np.array(data[cols])
Y = data['是否被引']

z=np.array(data2[cols2])
# 总数据条数 遍历显示
icount = X.shape[0]

for i in range(icount - 1):
    print(X[i], "--->", Y[i])

# 调用高斯朴素贝叶斯分类器
gnb = GaussianNB()

# 填入数据进行训练
gnb.fit(X, Y)

# 训练完后预测（此处用的测试数据是训练数据同一份，实际可以用新输入数据）
test_data = z
y_predicted, jll = gnb.predict(test_data)
fjll = open('../data/jll.txt','w')
fpredicted = open('../data/predicted.txt','w')
fjllno = open('../data/jllno.txt','w')
fjllyes = open('../data/jllyes.txt','w')

for J in jll:
    print(J,file=fjll,flush=True)
    print(J[0],file=fjllno,flush=True)
    print(J[1],file=fjllyes,flush=True)

for P in y_predicted:
    print(P,file=fpredicted,flush=True)
print(jll)
print(np.argmax(jll, axis=1))
print(y_predicted)

#显示预测结果
print("\n预测结果:\n", y_predicted)

#显示预测错误率
print("\n总数据%d条 预测失误%d条" % (datas.data.shape[0], (Y!= y_predicted).sum()))