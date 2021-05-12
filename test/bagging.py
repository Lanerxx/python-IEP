from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
# 导入数据集中的数据（每项有18个特征数据值，1个目标类别值）
datafile = 'D:/experience/临时实验/s1-100(纯特征整合版)(forone).xls' #参数初始化
data = pd.read_excel(datafile) #读取数据
score = []
# error = []
cols = [
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
        '近两年被引次数',
        '近两年被不同期刊引用次数',
        '近两年被不同国家引用次数',
        '近两年被不同机构引用次数',
        '近两年被不同学科引用次数',
        '早期被引频次（发表后两年的）',
        '早期被不同期刊引用次数',
        '早期被不同国家引用次数',
        '早期被不同机构引用次数',
        '早期被不同学科引用次数',

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
        '近两年被引次数',
        '近两年被不同期刊引用次数',
        '近两年被不同国家引用次数',
        '近两年被不同机构引用次数',
        '近两年被不同学科引用次数',
        '早期被引频次（发表后两年的）',
        '早期被不同期刊引用次数',
        '早期被不同国家引用次数',
        '早期被不同机构引用次数',
        '早期被不同学科引用次数',
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

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    clf.fit(train_X, train_Y)
    clf.score(test_X, test_Y)
    # predict_results = clf.predict(test_X)
    # print(accuracy_score(predict_results, test_Y))
    # conf_mat = confusion_matrix(test_Y, predict_results)
    # print(conf_mat)
    # print(classification_report(test_Y, predict_results))
    score.append(clf.score(test_X, test_Y))

print(score)
sum = 0
for s in score:
    sum += s
avg = sum/len(score)
print(avg)