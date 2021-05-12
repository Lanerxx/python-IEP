from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# 导入数据集中的数据（每项有18个特征数据值，1个目标类别值）
datafile = '../sourseData/GProjectDateTest1.xls' #参数初始化
data = pd.read_excel(datafile) #读取数据
col = data.columns.values.tolist()
#print(col)
datas=np.array(data)

# X为特征数据 y是类别值
cols = [
        '毕业学校等级',
        '外语水平',
        '学历',
        '期望薪资',
        '实习/工作经历',
        '项目经验',
        '性别',
        '专排百分比',
        '加权成绩',
        '期望岗位',
        '就业意向地',
        '简历被浏览次数',
        '论文数',
        '掌握技能',
        '专业技能证书',
        '获得荣誉/奖项',
        '五险一金',
        '定期体检',
        '年终奖',
        '带薪年假',
        '加班补助',
        '股票期权',
        '交通补贴',
        '住房补贴',
        '是愿意出差',
    ]
#print(cols)
X = np.array(data[cols])
print(X)
y = data['岗位是否发出邀请']


#递归特征消除法
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_
print('递归特征消除法')
for r in ranking:
        print(r)

# 逻辑回归
rfe2 = RFE(estimator=LogisticRegression(), n_features_to_select=1).fit(X, y)
print('逻辑回归')
for c in rfe2.ranking_:
    print(c)
