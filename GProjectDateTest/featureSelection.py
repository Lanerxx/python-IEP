from GProjectDateTest import reliefF
import numpy as np
import pandas as pd

datafile = '../sourseData/GProjectDateTest.xls' #参数初始化
data = pd.read_excel(datafile) #读取数据
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


X = np.array(data[cols])
y = data['岗位是否发出邀请']

score = reliefF.reliefF(X, y)
Fid = reliefF.feature_ranking(score)
print(Fid)
for f in Fid:
        print(f)
