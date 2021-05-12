import pandas as pd
import numpy as np

datafile = '../sourseData/s1-100(纯特征整合版)(forone).xls'

data = pd.read_excel(datafile)  # 读取数据
col = data.columns.values.tolist()
datas = np.array(data)
cols = col[2:]
X = np.array(data[cols])  # X为特征数据
Y = data['是否被引']  # Y是类别值
print(X)
print(Y)

features = ['施引文献',	'近两年被引次数',	'作者h指数',	'文献时间',	'rp',	'sjr',	'snip',	'两篇文章是否同一领域',	'是否为作者之前的论文',	'是否作者合著者的论文'	,
'文章作者是否作者之前引用过的作者',	'是否作者之前引用过的文章',	'标题相似性',	'摘要相似性',	'主题相似性',	'是否来源于美国',	'是否来源于欧洲',
'作者相似度（利用jaccard）',	'是否有资金赞助',	'早期被引频次（发表后两年的）',	'早期被不同期刊引用次数',	'早期被不同国家引用次数',
'早期被不同机构引用次数',	'早期被不同学科引用次数',	'近两年被不同期刊引用次数',	'近两年被不同国家引用次数',	'近两年被不同机构引用次数',	'近两年被不同学科引用次数'
]
# features = ['施引文献', '作者h指数', '引用文献时间间隔', 'SNIP', '两篇文章是否同一领域', '是否为作者之前的论文', '是否作者合著者的论文', '文章作者是否作者之前引用过的作者'
#     , '是否作者之前引用过的文章', '标题相似性', '摘要相似性', '主题相似性', '是否来源于美国', '是否来源于欧洲', '作者相似度（利用jaccard）', '是否有资金赞助', 'RP', 'SJR']
sumFeatures = len(features)  # 特征数
sumDates = len(Y)  # 数据数

# =============================1.求得各项特征1 / 0 均值点=============================
print("=============================1.求得各项特征1 / 0 均值点=============================")

aveFeatureOne_Zero = np.zeros([2, len(features)])

def calculateAveeFeatureOne_Zero(indexFeature):
    # 获取某项特征
    dataFeature = np.array(data[features[indexFeature]])
    # 获取1，0的数据
    data_1 = []
    data_0 = []
    for indexData in range(0, sumDates):
        if Y[indexData] == 1:
            data_1.append(dataFeature[indexData])
        else:
            data_0.append(dataFeature[indexData])
    # 计算1中均值
    sum = 0
    for i in data_1:
        sum += i
    aveFeatureOne_Zero[0][indexFeature] = sum / len(data_1)
    # 计算0中均值
    sum = 0
    for i in data_0:
        sum += i
    aveFeatureOne_Zero[1][indexFeature] = sum / len(data_0)

# 计算各项1，0均值
for indexFeature in range(0, sumFeatures):
    calculateAveeFeatureOne_Zero(indexFeature)
print(aveFeatureOne_Zero)

# =============================2.计算单个特征的的f1，作为初始权重范围=============================
print("=============================2.计算单个特征的的f1，作为初始权重范围=============================")

def cal_F1(precision, recall):
    if precision + recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0
    return f1

def cal_precision(target, predict, positive):
    fp = 0
    tp = 0
    length = len(target)
    for i in range(0, length):
        if target[i] == positive and predict[i] == positive:
            tp += 1
        elif target[i] != positive and predict[i] == positive:
            fp += 1
    # print("tp / fp : " + str(tp) + " / " + str(fp))
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    return precision

def cal_recall(target, predict, positive):
    tp = 0
    fn = 0
    length = len(target)
    for i in range(0, length):
        if target[i] == positive and predict[i] == positive:
            tp += 1
        elif target[i] == positive and predict[i] != positive:
            fn += 1
    # print("tp / fn : " + str(tp) + " / " + str(fn))
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    return recall

def F1(target, predict, positive):
    precision = cal_precision(target, predict, positive)
    recall = cal_recall(target, predict, positive)
    return cal_F1(precision, recall)

def weightDistant_scoreSingle(index):
    prediect = []
    # 测试每一项数据
    for row in range(0, sumDates):
        distant1 = 0
        distant0 = 0
        for indexFeature in range(0, sumFeatures):
            if indexFeature == index:
                distant1 += abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[0][indexFeature])
                distant0 += abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[1][indexFeature])
        if distant1 < distant0:
            prediect.append(1)
        else:
            prediect.append(0)
    return F1(Y, prediect, 1)

# 计算单个特征的影响值，作为初始权重
f1_features = []
indexFeature = 0
while indexFeature < sumFeatures:
    f1_features.append(weightDistant_scoreSingle(indexFeature))
    indexFeature += 1
print(f1_features)

# =============================3.利用带权重k临近迭，代比较距离后预测，对比f1，获得最终权重，排序=============================
print("=============================3.利用带权重曼哈顿迭代，比较距离后预测，对比f1，获得最终权重，排序=============================")

def weightDistant_score(index, weight):
    prediect = []
    # 测试每一项数据
    for row in range(0, sumDates):
        distant1 = 0
        distant0 = 0
        for indexFeature in range(0, sumFeatures):
            # 如果是正在测试的特征，距离权重使用传入值
            if indexFeature == index:
                distant1 += weight * abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[0][indexFeature])
                distant0 += weight * abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[1][indexFeature])
            # 如果非正在测试的特征，距离权重使用固定值
            else:
                distant1 += featureWeighted[indexFeature] * abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[0][indexFeature])
                distant0 += featureWeighted[indexFeature] * abs(
                    X[row][indexFeature] - aveFeatureOne_Zero[1][indexFeature])
            # print(features[indexFeature] + " 距离1：" + str(distant1) + " 距离0：" + str(distant0))
        # print("distant1:" + str(distant1) + "distant0:" + str(distant0))
        if distant1 < distant0:
            prediect.append(1)
        else:
            prediect.append(0)
    return F1(Y, prediect, 1)

featureWeighted = f1_features.copy()
# 循环迭代特征
wGap = 0.1 # 权重增幅（越小越精确，耗时越长）
wSum = 0 # 辅助查看程序进度
wCount = 0 # 辅助查看程序进度
indexFeature = 0
while indexFeature < sumFeatures:
    wSum += featureWeighted[indexFeature] + 0.5
    indexFeature += 1
wSum = wSum / wGap

indexFeature = 0
# 在不筛选特征的情况下的初始f1
initF1 = weightDistant_score(-1,0)
count = 3 # 迭代次数（越大越精确，耗时越长）
for i in range(0,count):
    print("==========----------第" + str(i + 1) + "次迭代----------==========")
    indexFeature = 0
    while indexFeature < sumFeatures:
        print("-----" + features[indexFeature] + "-----")
        w = 0
        while w < featureWeighted[indexFeature] + 0.5:
            culF1 = weightDistant_score(indexFeature, w)
            if initF1 < culF1:
                print("精度提升啦: initF1:" + str(initF1) + "culF1:" + str(culF1) + "当前权重：" + str(w))
                initF1 = culF1
                featureWeighted[indexFeature] = w
            w += wGap
            wCount += 1
            if wCount < wSum * count:
                print("当前进度：" + "%.2f%%" % ((wCount / wSum) / count *100))
            else:
                print("当前进度：99.99%")
        indexFeature += 1
    for i in featureWeighted:
        print(i)



