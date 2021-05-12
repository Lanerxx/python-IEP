from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# 导入数据集中的数据（每项有18个特征数据值，1个目标类别值）
datafile = '../sourseData/sno31-60(实验缺没被引近两年).xlsx' #参数初始化
data = pd.read_excel(datafile) #读取数据
score = []
# error = []
# ['施引文献', '作者h指数', 'snip', '两篇文章是否同一领域', '文献时间',
        # '是否为作者之前的论文', '是否作者合著者的论文', '文章作者是否作者之前引用过的作者', '是否作者之前引用过的文章',
        # '标题相似性', '摘要相似性', '主题相似性', '是否来源于美国', '是否来源于欧洲', '作者相似度（利用jaccard）', '是否有资金赞助', 'rp', 'sjr']
cols = [
        'sjr', '文章作者是否作者之前引用过的作者', '是否作者之前引用过的文章',
        '主题相似性', 'rp']
for i in range(31,61):
    data0 = data.loc[data['种子文献编号'] != i]
    data1 = data.loc[data['种子文献编号'] == i]
    # print(data0, data1)

    train_X = np.array(data0[cols])
    test_X = np.array(data1[cols])
    train_Y = data0['是否被引']
    test_Y = data1['是否被引']
    # print(train_X, train_Y, test_X, test_Y)

    gnb = GaussianNB()
    # bnb = BernoulliNB()
    # mnb = MultinomialNB()
    gnb.fit(train_X, train_Y)
    # bnb.fit(train_X, train_Y)
    # mnb.fit(train_X, train_Y)
    gnb.score(test_X, test_Y)
    # bnb.score(test_X, test_Y)
    # mnb.score(test_X, test_Y)

    # y_predicted = gnb.predict(test_X)
    # y_predicted_b = bnb.predict(test_X)
    # y_predicted_m = mnb.predict(test_X)

    # 显示预测结果
    # print("\n预测结果:\n", y_predicted)
    # print("\n预测结果:\n", y_predicted_b)
    # print("\n预测结果:\n", y_predicted_m)
    # # 显示预测错误率

    # print("\n总数据%d条 预测失误%d条" % (test_X.data.shape[0], (test_Y!= y_predicted).sum()))
    # print("\n总数据%d条 预测失误%d条" % (test_X.data.shape[0], (test_Y!= y_predicted_b).sum()))
    # print("\n总数据%d条 预测失误%d条" % (test_X.data.shape[0], (test_Y!= y_predicted_m).sum()))
    # #获取结果
    # print ('The Accuracy of GaussianNB Classifier is:', gnb.score(test_X, test_Y))
    # print (classification_report(test_Y, y_predicted, target_names = ['0', '1']))
    # print ('The Accuracy of BernoulliNB Classifier is:', bnb.score(test_X, test_Y))
    # print (classification_report(test_Y, y_predicted_b, target_names = ['0', '1']))
    # print ('The Accuracy of MultinomialNB Classifier is:', mnb.score(test_X, test_Y))
    # print (classification_report(test_Y, y_predicted_m, target_names = ['0', '1']))

    # error_num = (test_Y != y_predicted).sum()
    # error_percent = error_num/len(test_Y)
    # error.append(error_percent)
    score.append(gnb.score(test_X, test_Y))

print(score)
sum = 0
for s in score:
    sum += s
avg = sum/len(score)
print(avg)
