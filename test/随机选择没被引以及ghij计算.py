import os
import re
import  pandas  as pd
import math

# 参数初始化
datafile = '../sourseData/种子文献.xlsx'#种子文献文件位置
datafile1 = '../sourseData/26-50.xls'#被引文献文件位置
tofilename = '../data/s2018/26-50选取1v4.xls'#跑出来的文件的位置
findfile = '../sourseData/s2018/'#gh和ij外面一层文件夹
first_num = 26 #第一篇序号
last_num = 50 #最后一篇序号
mutiple_num = 4 #没被引相对被引的倍数
#如果不跑ghij就注释掉479-482行

def plural(word):
    if word.endswith('y'):
        return word[:-1]+'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh','ch']:
        return word+'es'
    elif word.endswith('an'):
        return word[-2:]+'en'
    elif word == '无' or word == '|':
        return word
    else:
        return word+'s'

def compute_cosine_topic(text_a, text_b):
    # 找单词及词频，单词的数量，和出现的次数。
    words1 = text_a.split(' ')
    words2 = text_b.split(' ')  # 分词
    words = []
    for word in words1:
        word = plural(word)
        words.append(word)
    words1.extend(words)
    words1 = list(set(words1))
    words = []
    for word in words2:
        word = plural(word)
        words.append(word)
    words2.extend(words)
    words2 = list(set(words2))
    return compute_cosine_common(words1, words2)

def compute_cosine(text_a, text_b):
    # 找单词及词频，单词的数量，和出现的次数。
    words1 = text_a.split(' ')
    words2 = text_b.split(' ')  # 分词
    return compute_cosine_common(words1, words2)

def compute_cosine_common(words1, words2):
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

def deal_G(k, df):
    find = findfile + 'GH/' + str(k)
    rootdir = find
    list = os.listdir(rootdir)
    datasum = []
    idsum = []
    # 将所有作者引用过的的论文放到一个列表里
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        if os.path.isfile(path):
            dfs = pd.read_excel(path)
            dfs = dfs.fillna(value=0)
            try:
                data = dfs['作者'].values
            except:
                data = dfs['Authors'].values
            try:
                id = dfs['作者 ID'].values
            except:
                id = dfs['Author(s) ID'].values
            # 将data的数据全部加到datasum后面,作者id整合到idsum列表里
            datasum.extend(data)
            idsum.extend(id)

    # print(len(datasum))
    # print(datasum)
    # print(idsum)

    # 对作者姓名进行处理分割,以，为分隔符
    author_list = []
    for data in datasum:
        author_list.extend(data.split(','))
    # print(len(author_list))
    # print(author_list)

    # 对作者id进行处理分割,以;为分隔符
    id_list = []
    for id in idsum:
        # print(id)
        id = str(id)
        if id != '0':
            id_list.extend(id.split(';'))
    # print(id_list)
    # print(len(id_list))

    # 进一步处理空值，得到种子文献作者id列表
    id_list1 = []
    for i in range(len(id_list)):
        if id_list[i] != '':
            id_list1.append(id_list[i])
    # print(id_list1)
    # print(len(id_list1))

    # 获取被引的作者名字列表以及作者id列表
    # k代表种子文献的标号,j代表论文编号
    author = df
    # print(len(author.values))
    for j in range(0, len(author.values)):
        # print(k)
        print(j)

        # authorname_list被引文献的作者形成的列表
        authored_list = []
        authored = df.loc[j, ['作者']]
        authored_list.extend(authored.values[0][0].split(','))
        # print(authored_list)
        # print(len(authored_list))

        # authorname_list被引文献的作者形成的列表
        ided_list = []
        authorid = df.loc[j, ['作者 ID']]
        # print(authorid.values[0][0])
        ided_list.extend(authorid.values[0][0].split(';'))
        # print(ided_list)

        # 进一步处理空值，得到被引文献作者id列表
        ided_list1 = []
        for i in range(len(ided_list)):
            if ided_list[i] != '':
                ided_list1.append(ided_list[i])
        # print(ided_list1)

        # 判断被引文献的作者名字列表是否在种子文献作者名字的列表里，是输出1，否输出0
        flag = False
        for authored in authored_list:
            # print(authored)
            authored = authored.strip()
            for author in author_list:
                # print(author)
                author = author.strip()
                if authored == author:
                    flag = True
                    # print(author)
                    # print(authored)
                    df.loc[j, ['文章作者是否作者之前引用过的作者']] = 1
                    # print(df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['文章作者是否作者之前引用过的作者']])
                    break
            if flag == True:
                break
                # df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['文章作者是否作者之前引用过的作者']] = 0

                # 如果作者姓名因为缩写出现没匹配，或者匹配失误，用作者id二次判断
            for ided in ided_list1:
                # print(ided)
                for id in id_list1:
                    # print(id)
                    if ided == id:
                        flag = True
                        df.loc[j, ['文章作者是否作者之前引用过的作者']] = 1
                        # print(df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['文章作者是否作者之前引用过的作者']])
                        break
                if flag == True:
                    break
                df.loc[j, ['文章作者是否作者之前引用过的作者']] = 0

def deal_H(k, df):
    find = findfile + 'GH/' + str(k)
    # print(find)
    rootdir = find
    list = os.listdir(rootdir)
    datasum = []
    # 将所有作者写过的论文放到一个列表里
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        if os.path.isfile(path):
            dfs = pd.read_excel(path)
            data = dfs['EID'].values
            # 将data的数据全部加到datasum后面
            datasum.extend(data)
    # print(len(datasum))
    # print(datasum)


    # 获取被引的EID
    # k代表种子文献的标号,j代表论文编号
    eids = df
    # print(len(eids.values))
    for j in range(0, len(eids.values)):
        # print(k)
        print(j)
        eids = df.loc[j, ['EID']]
        # print(eids.values[0])
        # 判断被引文献的EID是否在列表里，是输出1，否输出0
        for data in datasum:
            # print(eids.values[0][0])
            # print(data)
            if eids.values[0][0] == data:
                # print('*')
                df.loc[j, ['是否作者之前引用过的文章']] = 1
                # print(df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['是否为作者之前的论文']])
                break
            df.loc[j, ['是否作者之前引用过的文章']] = 0

def deal_I(k, df):
    find = findfile + 'IJ/' + str(k)
    # print(find)
    rootdir = find
    list = os.listdir(rootdir)
    datasum = []
    # 将所有作者写过的论文放到一个列表里
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        if os.path.isfile(path):
            dfs = pd.read_excel(path)
            data = dfs['EID'].values
            # 将data的数据全部加到datasum后面
            datasum.extend(data)
    # print(len(datasum))
    # print(datasum)


    # 获取被引的EID
    # k代表种子文献的标号,j代表论文编号

    eids = df
    # print(len(eids.values))
    for j in range(0, len(eids.values)):
        # print(k)
        print(j)
        eids = df.loc[j, ['EID']]
        # print(eids.values[0])
        # 判断被引文献的EID是否在列表里，是输出1，否输出0
        for data in datasum:
            # print(eids.values[0][0])
            # print(data)
            if eids.values[0][0] == data:
                # print('*')
                df.loc[j, ['是否为作者之前的论文']] = 1
                # print(df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['是否为作者之前的论文']])
                break
            df.loc[j, ['是否为作者之前的论文']] = 0

def deal_J(k, df, df_sort):
    find = findfile + 'IJ/' + str(k)
    rootdir = find
    list = os.listdir(rootdir)
    datasum = []
    idsum = []
    # 将所有作者合作者的论文放到一个列表里
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        if os.path.isfile(path):
            dfs = pd.read_excel(path)
            dfs = dfs.fillna(value=0)
            try:
                data = dfs['作者'].values
            except:
                data = dfs['Authors'].values
            # 将data的数据全部加到datasum后面
            datasum.extend(data)

    # print(len(datasum))
    # print(datasum)

    # 对合作者姓名进行处理分割,以，为分隔符
    author_list = []
    for data in datasum:
        author_list.extend(data.split(','))
    # print(len(author_list))
    # print(author_list)


    # 将合作者里的种子文献的作者删去
    author_sortlist = []
    author_sort = df_sort.loc[df_sort['种子文献编号'] == k, ['作者']]
    author_sortlist.extend(author_sort.values[0][0].split(','))
    # print(k)
    # print(author_sortlist)
    author_list1 = []

    for author in author_list:
        flag = False
        # 将作者名字前后的空格去掉
        author = author.strip()
        # print(author)
        for author_sort in author_sortlist:
            author_sort = author_sort.strip()
            # print(author_sort)
            # print(author_sort)
            if author == author_sort:
                flag = True
        if flag != True:
            author_list1.append(author)
    # print(author_list1)
    # print(len(author_list1))


    author = df
    # print(len(author.values))
    for j in range(0, len(author.values)):
        # print(k)
        print(j)

        # authorname_list被引文献的作者形成的列表
        authored_list = []
        authored = df.loc[j, ['作者']]
        authored_list.extend(authored.values[0][0].split(','))
        # print(authored_list)
        # print(len(authored_list))


        # flag用来跳出循环以及判断是否进入作者id循环
        # 判断被引文献的作者名字列表是否在种子文献作者名字的列表里，是输出1，否输出0
        flag = False
        for authored in authored_list:
            authored = authored.strip()
            # print(authored)
            for author in author_list1:
                # print(author)
                if authored == author:
                    flag = True
                    df.loc[j, ['是否作者合著者的论文']] = 1
                    # print(df.loc[(df['种子文献编号'] == k) & (df['论文序号'] == j), ['是否作者合著者的论文']])
                    break
            if flag == True:
                break
            df.loc[j, ['是否作者合著者的论文']] = 0

def similar(k, df, seed):
    topics = []
    titles = []
    abstracts = []
    authors = []

    topic1 = seed.loc[k, ['主题']].values
    topic2 = df['主题词'].values
    for i in topic2:
        for j in topic1:
            topic = compute_cosine_topic(i, j)
            topics.append(topic)

    title1 = seed.loc[k, ['标题']].values
    title2 = df['标题'].values
    for i in title2:
        for j in title1:
            title = compute_cosine(i, j)
            titles.append(title)

    abstract1 = seed.loc[k, ['摘要']].values
    abstract2 = df['摘要'].values
    for i in abstract2:
        for j in abstract1:
            abstract = compute_cosine(i, j)
            abstracts.append(abstract)

    author1 = seed.loc[k, ['作者']].values
    author2 = df['作者'].values
    for i in author2:
        for j in author1:
            author = compute_cosine(i, j)
            authors.append(author)

    df['标题相似性'] = titles
    df['摘要相似性'] = abstracts
    df['主题相似性'] = topics
    df['作者相似度（利用jaccard）'] = authors

if __name__ =='__main__':
    # 读取数据
    data = pd.read_excel(datafile)
    data1 = pd.read_excel(datafile1)
    col = data1.columns.values.tolist()
    dftotal = pd.DataFrame()
    for i in range(first_num, last_num + 1):
        # 被引dataframe
        df1 = data1[data1['种子文献编号'].isin([i])]
        # 没被引dataframe
        df0 = data1[~data1['种子文献编号'].isin([i])]
        df0 = pd.concat([df0, df1])
        df0 = pd.concat([df0, df1])
        df0 = df0.drop_duplicates(subset=['EID'], keep=False)
        # 与被引等长随机没被引dataframe
        df2 = df0.sample(n= mutiple_num * int(len(df1)))
        # 输出被引与没被引长度
        print(len(df1), len(df2), '\n')
        df1['是否被引'] = int(1)
        df2['是否被引'] = int(0)
        df = pd.concat([df1, df2])
        df = df.reset_index(drop=True)
        df['相对于第几篇种子'] = int(i)
        print(len(df))
        deal_G(i, df)
        deal_H(i, df)
        deal_I(i, df)
        deal_J(i, df, data)
        similar(i, df, data)
        dftotal = pd.concat([dftotal, df])
    dftotal.to_excel(tofilename)