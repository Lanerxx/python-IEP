from numpy import *
import xlrd

'''创建数据集：单词列表postingList, 所属类别classVec'''
def loadDataSet():
    postingList = []
    classVec = []
    #导入被引数据
    book1 = xlrd.open_workbook('data1.xls')
    sheet1 = book1.sheets()[0]
    nrows1 = sheet1.nrows
    for value in range(0,nrows1):
        postingList.append(sheet1.row_values(value))
        classVec.append(1)
    #导入为被引数据
    book0 = xlrd.open_workbook('data0.xls')
    sheet0 = book0.sheets()[0]
    nrows0 = sheet0.nrows
    for value in range(0, nrows0):
        postingList.append(sheet0.row_values(value))
        classVec.append(0)
    return postingList, classVec

'''获取所有单词的集合:返回不含重复元素的单词列表'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    # print(vocabSet)
    return list(vocabSet)

'''词集模型构建数据矩阵'''
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)
    # 遍历文档中的所有关键词，如果出现了词汇表中的关键词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词: %s 不在词汇表之中!" % word)
    # print(returnVec)
    return returnVec

'''朴素贝叶斯分类器训练函数'''
def _trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 文章数
    numWords = len(trainMatrix[0]) # 单词数
    # 被引文章的概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个被引文章，与文件的总数相除就得到了被引文章的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 构造单词出现次数列表
    p0Num = zeros(numWords) # [0,0,0,.....]
    p1Num = zeros(numWords) # [0,0,0,.....]
    p0Denom = 0.0;p1Denom = 0.0 # 整个数据集单词出现总数
    for i in range(numTrainDocs):
        # 遍历所有的文章，如果是被引用文件，就计算此被引文章中出现的关键词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #[0,1,1,....]->[0,1,1,...]
            p1Denom += sum(trainMatrix[i])
        else:
            # 如果不是被引文章，则计算未被引文章中出现的关键词的个数
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即被引文章的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现次数的占比
    p1Vect = p1Num / p1Denom# [1,2,3,5]/90->[1/90,...]
    # 类别0，即未被引文章的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在0类别下，每个单词出现次数的占比
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive

'''训练数据优化版本'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 总文件数
    numWords = len(trainMatrix[0]) # 总单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 被引文章的出现概率
    # 构造单词出现次数列表,p0Num 正常的统计,p1Num 被引的统计
    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    p0Num = ones(numWords)#[0,0......]->[1,1,1,1,1.....],ones初始化1的矩阵
    p1Num = ones(numWords)

    # 整个数据集单词出现总数，2.0根据样本实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整）
    # p0Denom 未被引的统计
    # p1Denom 被引的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]  # 累加关键词的频次
            p1Denom += sum(trainMatrix[i]) # 对每篇文章的关键词的频次 进行统计汇总
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即被引文章的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表,取对数避免下溢出或浮点舍入出错
    p1Vect = log(p1Num / p1Denom)
    # 类别0，即未被引文章的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

'''朴素贝叶斯算法'''
def testingNB():
    # 1. 加载数据集
    dataSet, Classlabels = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabList(dataSet)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in dataSet:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print('test',len(array(trainMat)[0]))
    # 4. 训练数据
    p0V, p1V, pAb = trainNB0(array(trainMat), array(Classlabels))
    # 5. 测试数据
    #换成键入数据
    testEntry = input()
    #testEntry = ['Computer', 'Psychology','Business']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类结果是: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = input()
    #testEntry = ['Health', 'Information', 'Education']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, '分类结果是: ', classifyNB(thisDoc, p0V, p1V, pAb))

testingNB()

