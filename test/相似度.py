import os
import re
import  pandas  as pd
import math

def compute_cosine(text_a, text_b):
    # 找单词及词频，单词的数量，和出现的次数。
    words1 = text_a.split(' ')
    words2 = text_b.split(' ')#分词
    words1_dict = {}
    words2_dict = {}
    for word in words1:
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()  #转换字符串中所有大写字符为小写
        if word != '' and word in words1_dict:
            num = words1_dict[word]
            words1_dict[word] = num + 1
        elif word != '':
            words1_dict[word] = 1
        else:
            continue
    for word in words2:
        # word = word.strip(",.?!;")
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()
        if word != '' and word in words2_dict:
            num = words2_dict[word]
            words2_dict[word] = num + 1
        elif word != '':
            words2_dict[word] = 1
        else:
            continue
    # print(words1_dict)
    # print(words2_dict)
    # return True
    dic1 = sorted(words1_dict.items(), key=lambda asd: asd[1], reverse=True)
    dic2 = sorted(words2_dict.items(), key=lambda asd: asd[1], reverse=True)
    # print(dic1)
    # print(dic2)

    # 得到词向量
    words_key = []
    for i in range(len(dic1)):
        words_key.append(dic1[i][0])  # 向数组中添加元素
    for i in range(len(dic2)):
        if dic2[i][0] in words_key:
            # print 'has_key', dic2[i][0]
            pass
        else:  # 合并
            words_key.append(dic2[i][0])
    # print(words_key)
    vect1 = []
    vect2 = []
    for word in words_key:
        if word in words1_dict:
            vect1.append(words1_dict[word])
        else:
            vect1.append(0)
        if word in words2_dict:
            vect2.append(words2_dict[word])
        else:
            vect2.append(0)
    # print(vect1)
    # print(vect2)

    # 计算余弦相似度
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(vect1)):
        sum += vect1[i] * vect2[i]
        sq1 += pow(vect1[i], 2)
        sq2 += pow(vect2[i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0
    return result

def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    grams_reference = reference.split(',')
    grams_model = model.split(',')
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient

if __name__ =='__main__':
    # 参数初始化

    # 种子文献
    datafile = '../sourseData/种子文献.xlsx'
    datafile1 = '../sourseData/sno31-60(similar).xls'
    toexcel = '../data/sno31-60.xlsx'
    # 读取数据
    seed = pd.read_excel(datafile)
    df = pd.read_excel(datafile1)
    df0 = pd.DataFrame()
    topics = []
    titles = []
    abstracts = []
    authors = []
    for k in range(1, 6):
        df1 = seed[seed['种子文献编号'].isin([k])]
        df2 = df[df['种子文献编号'].isin([k])]
        df0 = pd.concat([df0, df2])
        topic1 = df1['主题'].values
        topic2 = df2['主题词'].values
        for i in topic2:
            for j in topic1:
                topic = compute_cosine(i, j)
                topics.append(topic)

        title1 = df1['标题'].values
        title2 = df2['标题'].values
        for i in title2:
            for j in title1:
                title = compute_cosine(i, j)
                titles.append(title)

        abstract1 = df1['摘要'].values
        abstract2 = df2['摘要'].values
        for i in abstract2:
            for j in abstract1:
                abstract = compute_cosine(i, j)
                abstracts.append(abstract)

        author1 = df1['作者'].values
        author2 = df2['作者'].values
        for i in author2:
            for j in author1:
                author = compute_cosine(i, j)
                authors.append(author)

    #df0['标题相似性'] = titles
    #df0['摘要相似性'] = abstracts
    df0['主题相似性'] = topics
    df0['作者相似度（利用jaccard）'] = authors
    df0.to_excel(toexcel)
