from math import log
import operator


# 计算给定数据集的香农熵（通用）
def calcShannonEnt(dataSet):
    # 为所有可能的分类创建"类别：频数"字典
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        
    # 计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 示例数据：海洋生物鉴定
def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    
    return dataSet, labels


# 按给定特征划分数据集（通用）
# axis：划分特征，数据集中第(axis+1)列
# value: 划分特征值
# retDataSet：返回axis特征等于value的样例（不包含axis特征值）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    
    return retDataSet


# 根据信息增益的原理，选择最好的数据集划分方式（通用）
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):
        # 列表推导式：将dataSet中第i列的特征值储存到一个新的列表中
        featList = [example[i] for example in dataSet]
        # set(featList):取featList中所有的值（不重复）
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 对同一特征的多个值求熵，按概率加权求和。
        # 得到按该特征划分的条件熵及信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


# 多数表决，决定该分支的分类（通用）
def majorityCnt(classList):
    classCount={}
    
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        
        classCount[vote] += 1
    
    sortedClassCount = sorted(classCount.items(),\
                              key=operator.itemgetter(1),\
                              reverse=True)
    
    return sortedClassCount[0][0]


# 递归构建决策树（通用）
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征，返回出现最多的类作为该分支的类，停止
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
        
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                                        (dataSet, bestFeat, value),\
                                        subLables)
    
    return myTree


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)