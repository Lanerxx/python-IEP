from sklearn.svm import SVC
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit

datafile = '../sourseData/s1-100(纯特征整合版+早近合一)(forone)(分类器).xls' #参数初始化
data = pd.read_excel(datafile) #读取数据
score = []
# error = []
cols = [
    # '标题相似性',
    #     '摘要相似性',
    #     '主题相似性',
    #     '作者相似度（利用jaccard）',
    #     '是否为作者之前的论文',
    #     '是否作者合著者的论文',
    #     '文章作者是否作者之前引用过的作者',
    #     '是否作者之前引用过的文章',
    #     '施引文献',
    #     '作者h指数',
    #     '两篇文章是否同一领域',
    #     '文献时间',
    #     '是否来源于美国',
    #     '是否来源于欧洲',
    #     '是否有资金赞助',
    #     'rp',
    #     'sjr',
    #     'snip',
    #     '近两年被引次数',
    #     '近两年被不同期刊引用次数',
    #     '近两年被不同国家引用次数',
    #     '近两年被不同机构引用次数',
    #     '近两年被不同学科引用次数',
    #     '早期被引频次（发表后两年的）',
    #     '早期被不同期刊引用次数',
    #     '早期被不同国家引用次数',
    #     '早期被不同机构引用次数',
    #     '早期被不同学科引用次数',
    '是否来源于美国',
    '文献时间',
    '是否来源于欧洲',
    '文章作者是否作者之前引用过的作者',
    'snip',
    '是否为作者之前的论文',
    '近期引用状况',
    'rp',
    '早期引用状况',
    '是否有资金赞助',
    'sjr',
    '主题相似性',
    '标题相似性',
    '摘要相似性',
    '是否作者之前引用过的文章',
]
all_cols = [
        '标题相似性',
        '摘要相似性',
        '主题相似性',
        '作者相似度（利用jaccard）',
        '是否为作者之前的论文',
        '是否作者合著者的论文',
        '文章作者是否作者之前引用过的作者',
        '是否作者之前引用过的文章',
        '施引文献',
        '作者h指数',
        '两篇文章是否同一领域',
        '文献时间',
        '是否来源于美国',
        '是否来源于欧洲',
        '是否有资金赞助',
        'rp',
        'sjr',
        'snip',
        '近两引用状况',
        '早期引用状况',
]
for i in range(1,101):
    data0 = data.loc[data['种子文献编号'] != i]
    data1 = data.loc[data['种子文献编号'] == i]
    # print(data0, data1)
    print(i)
    train_X = np.array(data0[cols])
    test_X = np.array(data1[cols])
    train_Y = data0['是否被引']
    test_Y = data1['是否被引']
    # print(train_X, train_Y, test_X, test_Y)

    model = SVC(kernel='linear', probability=True)
    model.fit(train_X,train_Y)
    score.append(model.score(test_X, test_Y))

print(score)
sum = 0
for s in score:
    sum += s
avg = sum / len(score)
print(avg)
