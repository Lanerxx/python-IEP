import matplotlib.pyplot as plt
import numpy as np
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

x = ['作者相似度（利用jaccard）',
'施引文献',
'是否作者合著者的论文',
'是否来源于美国',
'文献时间',
'是否来源于欧洲',
'文章作者是否作者之前引用过的作者',
'是否为作者之前的论文',
'近期引用状况',
'早期引用状况',
'是否有资金赞助',
'主题相似性',
'标题相似性',
'摘要相似性',
'是否作者之前引用过的文章',
]
y = [0.883107637,
0.884297263,
0.883875594,
0.886529688,
0.886706236,
0.886137396,
0.885274834,
0.884427436,
0.880383114,
0.878665957,
0.868253527,
0.865083364,
0.850671576,
0.8483411,
0.841268995,
]

plt.figure()
plt.plot(x, y, color='black', linewidth=2.0, linestyle='-')
plt.xlabel('Feature')
plt.ylabel('Accuracy of classifier')
plt.legend(labels=['Line-a'],  loc='best')
plt.title("A graph of the classifier's accuracy as features are removed")
plt.show()

# 如果想要调整x,y坐标轴的刻度
# plt.xticks(new_ticks)
# plt.yticks([1, 2, 3],['bad', 'normal', 'good'])