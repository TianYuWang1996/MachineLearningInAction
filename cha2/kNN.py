from numpy import *
import operator
from os import listdir


# 创建数据集和标签向量（示例）
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


# 构建分类器（通用）
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    # argsort()函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    
    # 提取距离最近的k个对象的标签，并计算各个标签的频数，以字典的形式存储
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    
    # 对标签频数字典，以频数从大到小的方式排序。
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)
    # 返回频数最大的标签
    return sortedClassCount[0][0]


# 将数据的文本文件转换为数据集和标签向量（约会网站）
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    
    return returnMat, classLabelVector


# 归一化特征值（通用）
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    
    return normDataSet, ranges, minVals


# 对于分类器的测试（约会网站）
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                           datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))


# 分类器的使用——预测函数（约会网站）
def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])


# 将32*32的二进制图像矩阵转化为1*1024的向量（数字识别）
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# 对于分类器的测试（数字识别）
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/%s" %fileNameStr)
    
    testFileList = listdir("testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s" %fileNameStr)
        
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              %(classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" %errorCount)
    print("\nthe total error rate is: %f" %(errorCount/float(mTest)))