from numpy import *

# 读取文件数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        # 将curLine的每个元素映射为浮点型
        #fltLine = map(float, curLine)
        fltLine = []
        for line in curLine:
            fltLine.append(float(line))
        dataMat.append(fltLine)
    return dataMat


# 将数据集就某特征值的大小进行划分
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 返回数据集标签的均值
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

# 返回数据集标签的总方差
def regErr(dataSet):
    return var(dataSet[:, -1])*shape(dataSet)[0]


# 寻找总方差最小的划分方式（特征，值）
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]  # 容许的误差下降值 1 
    tolN = ops[1]  # 切分最少样本数 4
    
    # 数据集标签值不重复，则无法划分
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    
    m, n =shape(dataSet)
    S = errType(dataSet)
    
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        feat_tuple = []
        for i in range(m):
            feat_tuple.append(dataSet[i, featIndex])
        feat_tuple = tuple(feat_tuple)
        for splitVal in set(feat_tuple):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 划分子矩阵的样本数不得小于4，否则跳过该值
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    
    # 若最佳划分带来的误差下降不大，则没必要划分
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    
    return bestIndex, bestValue


# CART树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 找到划分特征及值
    feat, val =chooseBestSplit(dataSet, leafType, errType, ops)
    # 递归终止条件：节点不可再分，返回单值（回归树）或线性模型（模型树）
    if feat == None:
        return val
    # 创建节点
    retTree = {}
    retTree["spInd"] = feat
    retTree["spVal"] = val
    # 划分左右分支
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree["left"] = createTree(lSet, leafType, errType, ops)
    retTree["right"] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断是否为树（字典）
def isTree(obj):
    return (type(obj).__name__=="dict")


# 对树进行塌陷处理：返回树平均值
def getMean(tree):
    if isTree(tree["right"]):
        tree["right"] = getMean(tree["right"])
    if isTree(tree["left"]):
        tree["left"] = getMean(tree["left"])
    
    return (tree["left"]+tree["right"])/2.0


# 回归树剪枝函数
def prune(tree, testData):
    
    # 没有测试数据则对树进行塌陷处理
    if shape(testData)[0] == 0:
        return getMean(tree)
    
    # 当树的左右分支有一个为子树时，按树的特征值划分测试数据
    if (isTree(tree["right"]) or isTree(tree["left"])):
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
    # 左分支为子树时，递归调用剪枝函数
    if isTree(tree["left"]):
        tree["left"] = prune(tree["left"], lSet)
    # 右分支为子树时，递归调用剪枝函数
    if isTree(tree["right"]):
        tree["right"] = prune(tree["right"], rSet)
    
    # 从叶节点开始剪枝
    if not isTree(tree["left"]) and not isTree(tree["right"]):
        # 划分左右分支
        lSet, rSet = binSplitDataSet(testData, tree["spInd"], tree["spVal"])
        # 合并前误差
        errorNoMerge = sum(power(lSet[:, -1] - tree["left"], 2)) + sum(power(rSet[:, -1] - tree["right"], 2))
            
        # 合并后误差
        treeMean = (tree["left"]+tree["right"])/2.0
        errorMerge = sum(power(testData[:, -1]- treeMean, 2))
        
        # 对合并前后的误差比较
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree  # 剪枝完成


# 模型树参数 modelLeaf & modelErr
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError("This matrix is singluar, cannot do inverse, try increase the second value of ops")
    ws = xTx.I*(X.T*Y)
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat, 2))


def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n+1)))
    X[:, 1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree["spInd"]] > tree["spVal"]:
        if isTree(tree["left"]):
            return treeForeCast(tree["left"], inData, modelEval)
        else:
            return modelEval(tree["left"], inData)
    else:
        if isTree(tree["right"]):
            return treeForeCast(tree["right"], inData, modelEval)
        else:
            return modelEval(tree["right"], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat