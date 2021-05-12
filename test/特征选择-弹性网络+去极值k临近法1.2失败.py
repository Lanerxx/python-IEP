from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.metrics import roc_curve, auc

datafile = '../sourseData/31-60(newnocity1v4forone).xlsx'

data = pd.read_excel(datafile)  # 读取数据
col = data.columns.values.tolist()
datas = np.array(data)
cols = col[2:]
X = np.array(data[cols]) # X为特征数据
Y = data['是否被引'] # Y是类别值
print(X)
print(Y)

features = ['施引文献','作者h指数','引用文献时间间隔','SNIP','两篇文章是否同一领域','是否为作者之前的论文','是否作者合著者的论文','文章作者是否作者之前引用过的作者'
            ,'是否作者之前引用过的文章','标题相似性','摘要相似性','主题相似性','是否来源于美国','是否来源于欧洲','作者相似度（利用jaccard）','是否有资金赞助','RP','SJR']
sumFeatures = len(features) # 特征数
sumDates = len(Y) # 数据数
featureWeight = np.zeros(sumFeatures) # 特征权重
for i in range(1, sumFeatures):
    featureWeight[i] = 0.1

# =============================1.弹性网络，初步求得特征影响程度，作为k临近权重的迭代范围=============================
print("=============================1.弹性网络，初步求得特征影响程度，作为k临近权重的迭代范围=============================")
# 弹性网络，初步求得特征影响程度，作为k临近权重的迭代范围
def elastic(Xtest):
    # 以20%的数据构建测试样本，剩余作为训练样本
    # 这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
    X_train,X_test,y_train,y_test=train_test_split(Xtest,Y,test_size=0.3,random_state =1)
    elastic1= linear_model.ElasticNet(alpha=0.1,l1_ratio=0.2)  # 设置lambda值,l1_ratio值
    elastic1.fit(X_train,y_train)  #使用训练数据进行参数求解
    y_hat3 = elastic1.predict(X_test)  #对测试集的预测
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_hat3))
    return RMSE
# 求得整体特征的RMSE（均方根误差亦称标准误差），即RMSE越小越好
sumRMSE = elastic(X)
# 循环求得删去单个特征的RMSE，与sumRMSE做比较。RMSE比与sumRMSE相比较越大，则该特征对预测准确更有用
for indexFeature in range(0,sumFeatures):
    Xtemp = X.copy()
    Xtest = np.delete(Xtemp, indexFeature, axis=1)
    RMSE = elastic(Xtest)
    featureWeight[indexFeature] = featureWeight[indexFeature] + (round(RMSE - sumRMSE, 5))*100
print(featureWeight)

# =============================2.四方位去极值，求得各项特征1 / 0 均值点=============================
print("=============================2.四方位去极值，求得各项特征1 / 0 均值点=============================")

aveFeatureOne_Zero = np.zeros([2,len(features)])

# 四分法寻找孤立点界限
def quartileMethod(data):
    quartile1 = np.percentile(data, (25))
    quartile3 = np.percentile(data, (75))
    # print('最小数:' + str(np.percentile(data, (1))) + ' / 最大数:' + str(np.percentile(data, (99))))
    # print('第1四分位数:' + str(quartile1) + ' / 第2四分位数:' + str(quartile2) + ' / 第3四分位数:' + str(quartile3))
    cutUp = quartile3 + (quartile3 - quartile1) * 1.5
    cutDown = quartile1 - (quartile3 - quartile1) * 1.5
    # print('孤立点上界：' + str(cutUp) + ' / 孤立点下界：' + str(cutDown))
    return cutUp,cutDown

def calculateAveeFeatureOne_Zero(indexFeature):
    # 获取某项特征
    dataFeature = np.array(data[ features[ indexFeature ] ])
    # 获取1，0的数据
    data_1 = [ ]
    data_0 = [ ]
    data_one = [ ]
    data_zero = [ ]
    for indexData in range(0, sumDates):
        if Y[ indexData ] == 1:
            data_1.append(dataFeature[ indexData ])
        else:
            data_0.append(dataFeature[ indexData ])
    # 筛选1中的孤立值
    cutUp_1, cutDown_1 = quartileMethod(data_1)
    for d1 in data_1:
        if d1 <= cutUp_1 and d1 >= cutDown_1:
            data_one.append(d1)

    # 筛选0中的孤立值
    cutUp_0, cutDown_0 = quartileMethod(data_0)
    for d0 in data_0:
        if d0 <= cutUp_0 and d0 >= cutDown_0:
            data_zero.append(d0)
    # 计算1中均值
    sum = 0
    for i in data_one:
        sum += i
    # print("均值为：", sum / len(data_one))
    aveFeatureOne_Zero[ 0 ][ indexFeature ] = sum / len(data_one)
    # 计算0中均值
    sum = 0
    for i in data_zero:
        sum += i
    # print("均值为：", sum / len(data_zero))
    aveFeatureOne_Zero[ 1 ][ indexFeature ] = sum / len(data_zero)

# 计算各项1，0均值
for indexFeature in range(0,sumFeatures):
    calculateAveeFeatureOne_Zero(indexFeature)
print(aveFeatureOne_Zero)

# =============================3.利用带权重k临近迭代比较距离后预测，对比auc，获得最终权重，排序=============================
print("=============================3.利用带权重k临近迭代比较距离后预测，对比auc，获得最终权重，排序=============================")

def weightDistant_scoreInit(index,weight):
    prediect = []
    # 测试每一项数据
    for row in range(0,sumDates):
        distant1 = 0
        distant0 = 0
        for indexFeature in range(0,sumFeatures):
            # 如果是正在测试的特征，距离权重使用传入值
            if indexFeature == index:
                distant1 += weight * abs(
                    X[ row ][ indexFeature ] - aveFeatureOne_Zero[0][ indexFeature ])
                distant0 += weight * abs(
                    X[ row ][ indexFeature ] - aveFeatureOne_Zero[1][ indexFeature ])
            # 如果非正在测试的特征，距离权重使用固定值
            else:
                distant1 += 0.01*abs(X[row][indexFeature] - aveFeatureOne_Zero[0][indexFeature])
                distant0 += 0.01*abs(X[row][indexFeature] - aveFeatureOne_Zero[1][indexFeature])
            # print(features[indexFeature] + " 距离1：" + str(distant1) + " 距离0：" + str(distant0))
        # print("distant1:" + str(distant1) + "distant0:" + str(distant0))
        if distant1 < distant0:
            prediect.append(1)
        else:
            prediect.append(0)
    fpr, tpr, thresholds = roc_curve(Y, prediect, pos_label=1)
    return auc(fpr, tpr)
def weightDistant_score(index,weight):
    prediect = []
    # 测试每一项数据
    for row in range(0,sumDates):
        distant1 = 0
        distant0 = 0
        for indexFeature in range(0,sumFeatures):
            # 如果是正在测试的特征，距离权重使用传入值
            if indexFeature == index:
                distant1 += weight * abs(
                    X[ row ][ indexFeature ] - aveFeatureOne_Zero[0][ indexFeature ])
                distant0 += weight * abs(
                    X[ row ][ indexFeature ] - aveFeatureOne_Zero[1][ indexFeature ])
            # 如果非正在测试的特征，距离权重使用固定值
            else:
                distant1 += featureWeighted[indexFeature]*abs(X[row][indexFeature] - aveFeatureOne_Zero[0][indexFeature])
                distant0 += featureWeighted[indexFeature]*abs(X[row][indexFeature] - aveFeatureOne_Zero[1][indexFeature])
            # print(features[indexFeature] + " 距离1：" + str(distant1) + " 距离0：" + str(distant0))
        # print("distant1:" + str(distant1) + "distant0:" + str(distant0))
        if distant1 < distant0:
            prediect.append(1)
        else:
            prediect.append(0)
    fpr, tpr, thresholds = roc_curve(Y, prediect, pos_label=1)
    return auc(fpr, tpr)

# 初始auc
initAuc  = 0
print(initAuc)
# 首次迭代特征
indexFeature = 0
featureWeighted = featureWeight.copy()
while indexFeature < sumFeatures:
    print(features[indexFeature])
    initAuc = 0
    w = 0
    while w < featureWeight[ indexFeature ] + 2:
        culAuc = weightDistant_scoreInit(indexFeature, w)
        if initAuc < culAuc:
            print("精度提升啦: ininAuc:" + str(initAuc) + "culAuc:" + str(culAuc) + "当前权重：" + str(w))
            initAuc = culAuc
            featureWeighted[indexFeature] = w
        w += 0.01
    indexFeature += 1
for i in  featureWeighted:
    print(i)

print("================n====================")

# 循环迭代特征
indexFeature = 0
featureWeighted = featureWeight.copy()
while indexFeature < sumFeatures:
    print("============")
    initAuc = 0
    w = 0
    while w < featureWeight[ indexFeature ] + 2:
        culAuc = weightDistant_score(indexFeature, w)
        if initAuc < culAuc:
            print("精度提升啦: ininAuc:" + str(initAuc) + "culAuc:" + str(culAuc) + "当前权重：" + str(w))
            initAuc = culAuc
            featureWeighted[indexFeature] = w
        w += 0.05
    indexFeature += 1
for i in  featureWeighted:
    print(i)
