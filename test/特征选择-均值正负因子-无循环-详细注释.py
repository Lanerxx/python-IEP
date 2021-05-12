import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc

datafile = '../sourseData/31-60(newnocity1v4forone).xlsx'
data = pd.read_excel(datafile)  # 读取数据
col = data.columns.values.tolist()
datas = np.array(data)

# X为特征数据 y是类别值
cols = col[1:]
# print(cols)
X = np.array(data[cols])
Y = data['是否被引']
print(X)
print(Y)

features = ['施引文献','作者h指数','引用文献时间间隔','SNIP','两篇文章是否同一领域','是否为作者之前的论文','是否作者合著者的论文','文章作者是否作者之前引用过的作者'
            ,'是否作者之前引用过的文章','标题相似性','摘要相似性','主题相似性','是否来源于美国','是否来源于欧洲','作者相似度（利用jaccard）','是否有资金赞助','RP','SJR']
sumFeatures = len(features)
sumDates = len(Y)
print("特征数：" + str(sumFeatures))
print("数据数：" + str(sumDates))

def laner_score(featuresCut):
    sumFeatures = len(featuresCut)
    # 特征总均值
    aveFeatures = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # 特征1均值
    ave_1_Features = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
    # 特征总均值与1均值的正负关系
    compareFeatures = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]

    print("============1:计算============")
    indexFeatures = 0
    # 遍历列
    while indexFeatures < sumFeatures:
        print("特征：" + features[indexFeatures])
        # 获取某一列特征
        dataFeature = np.array(data[features[indexFeatures]])

        # 求均值
        sum = 0
        for i in dataFeature:
            sum = sum + i
        # 均值存入数组
        aveFeatures[indexFeatures] = sum / sumDates
        print("特征总均值：")
        print(sum / sumDates)

        indexData = 0
        # 是1的该特征的总值
        sum1 = 0
        # 1总共多少个
        sum1Datas = 0
        while indexData < sumDates:
            if(Y[indexData]==1):
                sum1 = dataFeature[indexData] + sum1
                sum1Datas = sum1Datas + 1
            indexData = indexData + 1
        # 均值存入_1_数组
        ave_1_Features[indexFeatures] = sum1 / sum1Datas
        print("1的均值")
        print(sum1 / sum1Datas)

        # 特征总均值与1均值的正负关系
        if ave_1_Features[indexFeatures] > aveFeatures[indexFeatures]:
            compareFeatures[indexFeatures] = 1
        else:
            compareFeatures[indexFeatures] = -1

        indexFeatures = indexFeatures + 1
        print('--------------------------------')

    print("============1:结果============")
    print("特征总均值：")
    print(aveFeatures)
    print("特征 1 均值：")
    print(ave_1_Features)
    print("特征正负关系：")
    print(compareFeatures)
    print(features)

    print("============2:计算============")
    # 预测结果
    predicrtConclusion = []
    # 初步根据均值验证
    # 遍历每行数据
    indexData = 0
    for rowX in X:
        sumConclusion = 0
        i = 0
        # 遍历每个特征
        while i < sumFeatures:
            # 计算该均值是否大于该特征总均值，并乘以正负系数，然后加入结论值
            sumConclusion = sumConclusion + (rowX[i]-aveFeatures[i])*compareFeatures[i]
            i = i + 1

        if sumConclusion > 0:
            predicrtConclusion.append(1)
        else:
            predicrtConclusion.append(0)
        indexData = indexData +1
    print("============2:结果============")
    print(predicrtConclusion)

    print("============3:对比============")
    indexData = 0
    # 1预测正确率
    ala1_1 = 0
    ala1_1Count = 0
    ala1Count = 0
    ala1_Count = 0
    # 0预测正确率
    ala0_0 = 0
    ala0_0Count = 0
    ala0Count = 0
    ala0_Count = 0
    # 总正确率
    alaA_A = 0
    alaA_ACount = 0
    while indexData < sumDates:
        print("实际数据：" + str(Y[indexData]) + "预测数据：" + str(predicrtConclusion[indexData]))
        if Y[indexData] == predicrtConclusion[indexData]:
            alaA_ACount = alaA_ACount + 1
            print("正确")
            if Y[indexData] == 1:
                ala1_1Count = ala1_1Count + 1
            else:
                ala0_0Count = ala0_0Count + 1

        # 计算正确的 1 ，0 分别有多少
        if Y[indexData] == 1:
            ala1Count = ala1Count + 1
        else:
            ala0Count = ala0Count + 1

        # 计算预测的 1 ，0 分别有多少
        if predicrtConclusion[indexData] == 1:
            ala1_Count = ala1_Count + 1
        else:
            ala0_Count = ala0_Count + 1

        indexData = indexData + 1

    ala1_1 = ala1_1Count / ala1Count
    ala0_0 = ala0_0Count / ala0Count
    alaA_A = alaA_ACount / sumDates

    print("总正确率：" + str(alaA_A))
    print("1 正确率：" + str(ala1_1))
    print("1 应有：" + str(ala1Count) + "  预测了：" + str(ala1_Count))
    print("0 正确率：" + str(ala0_0))
    print("0 应有：" + str(ala0Count) + "  预测了：" + str(ala0_Count))

    fpr, tpr, thresholds = roc_curve(Y, predicrtConclusion, pos_label=1)
    print("-----sklearn:",auc(fpr, tpr))

indexFeatures = 0
while indexFeatures < sumFeatures:
    featuresCut = features.copy()
    del featuresCut[indexFeatures]
    print(featuresCut)
    laner_score(featuresCut)
    print("==================================LANER===================================")
    indexFeatures = indexFeatures + 1










