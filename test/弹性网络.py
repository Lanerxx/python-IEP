from sklearn.model_selection import train_test_split
import pandas  as pd
import numpy  as np
from sklearn import linear_model, metrics

datafile = '../sourseData/31-60(newnocity1v4forone)的副本.xlsx'

data = pd.read_excel(datafile)  # 读取数据
col = data.columns.values.tolist()
datas = np.array(data)
# X为特征数据 Y是类别值
cols = col[2:]
# print(cols)
X = np.array(data[cols])
Y = data['是否被引']
print(X)
print(Y)


features = ['施引文献','作者h指数','引用文献时间间隔','SNIP','两篇文章是否同一领域','是否为作者之前的论文','是否作者合著者的论文','文章作者是否作者之前引用过的作者'
            ,'是否作者之前引用过的文章','标题相似性','摘要相似性','主题相似性','是否来源于美国','是否来源于欧洲','作者相似度（利用jaccard）','是否有资金赞助','RP','SJR']
sumFeatures = len(features)
sumDates = len(Y)
featureWeight = np.ones(sumFeatures)
print("特征数：" + str(sumFeatures))
print("数据数：" + str(sumDates))

def elastic(Xtest):
    # 以20%的数据构建测试样本，剩余作为训练样本
    # 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
    X_train,X_test,y_train,y_test=train_test_split(Xtest,Y,test_size=0.3,random_state =1)
    elastic1= linear_model.ElasticNet(alpha=0.1,l1_ratio=0.2)  # 设置lambda值,l1_ratio值
    elastic1.fit(X_train,y_train)  #使用训练数据进行参数求解
    y_hat3 = elastic1.predict(X_test)  #对测试集的预测
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_hat3))
    return RMSE

sumRMSE = elastic(X)
for indexFeature in range(0,sumFeatures):
    print(features[indexFeature])
    Xtemp = X.copy()
    Xtest = np.delete(Xtemp, indexFeature, axis=1)
    RMSE = elastic(Xtest)
    featureWeight[indexFeature] += (round(RMSE - sumRMSE, 5))*100
print(featureWeight)



