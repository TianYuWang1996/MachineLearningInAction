from numpy import *


# 示例数据，返回切分好的词条集合及类别标签集合
def loadDataSet():
    postingList = [["my", "dog", "has", "flea", "problems", "help", "please"],
                   ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
                   ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
                   ["stop", "posting", "stupid", "worthless", "garbage"],
                   ["mr", "lick", "ate", "my", "steak", "how", "to", "stop", "him"],
                   ["quit", "buying", "worthless", "dog", "food", "stupid"]]
    classVec = [0,1,0,1,0,1]  # 1代表侮辱性文字， 0代表正常言论
    
    return postingList, classVec


# 提取特征：创建一个包含文档里不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    
    return list(vocabSet)


# 创建inputSet的特征向量（基于词集模型）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    
    return returnVec


# 创建inputSet的特征向量（基于词袋模型）
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    
    return returnVec

# 朴素贝叶斯分类器训练函数（二分类问题）
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) #文本数
    numWords = len(trainMatrix[0])  #第一个文本的单词数
    # 计算trainCategory（0，1）中类别为1的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    
    # 初始化参数
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    
    # 遍历所有文本，更新参数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = log(p1Num/p1Denom)  # 取对数，防止下溢出
    p0Vect = log(p0Num/p0Denom)
    
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数（二分类问题）
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
    p1 = sum(vec2Classify * p1Vect) + log(pClass1)
    p0 = sum(vec2Classify * p0Vect) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 测试函数（使用示例数据）
def testingNB():
    # 准备数据：词表、训练特征矩阵
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    
    # 训练分类器：获得pAb、(1-pAb)、p1V、p0V
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    
    # 基于测试文本的朴素贝叶斯分类
    testEntry = ["love", "my", "dalmation"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    
    testEntry = ["stupid", "garbage"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))


# 文件解析函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


# 测试函数（垃圾邮件/交叉验证）
def spamTest():
    
    docList=[]
    classList=[]
    fullText=[]
    
    # 导入并解析文本文件
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = textParse(open("email/ham/%d.txt" %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    # 构建不重复词表
    vocabList = createVocabList(docList)
    
    # 随机构建测试集及训练集
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    # 利用随机生成的训练集训练分类器
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error: ",docList[docIndex])
            
    print("the error rate is : ", float(errorCount)/len(testSet))


# 计算文本中词条出现的频次，排序，返回前30名
def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1),\
                        reverse=True)
    
    return sortedFreq[:30]

# RSS源分类器的测试：输出错误率
def localWords(feed1, feed0):
    import feedparser
    
    docList = []
    classList = []
    fullText = []
    
    minLen = min(len(feed1["entries"]), len(feed0["entries"]))
    
    for i in range(minLen):
        wordList = textParse(feed1["entries"][i]["summary"])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = textParse(feed0["entries"][i]["summary"])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error: ",docList[docIndex])
            
    print("the error rate is: ", float(errorCount)/len(testSet))
    
    return vocabList, p0V, p1V


# RSS源分类器的测试：显示地域相关用词
def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topSF.append((vocabList[i], p1V[i]))
    
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])