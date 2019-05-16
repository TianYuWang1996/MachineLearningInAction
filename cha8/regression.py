from numpy import *


# 加载文件中的数据，返回特征值向量和标签向量
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(curLine[-1])
    
    return dataMat, labelMat


# 标准回归函数，返回特征矩阵
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  # 若矩阵xTx行列式为0
        # 奇异矩阵不能求逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


# 绘制数据散点图和回归拟合直线
def standRegresPlot(xArr, yArr, ws):
    import matplotlib.pyplot as plt
    
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float)
    yHat = xMat*ws
    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],\
               yMat.T[:, 0].flatten().A[0])
    # 绘制拟合直线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1], yHat)
    
    plt.show()


# 检查线性回归模型拟合情况，打印预测值与实际值的相关系数
def checkRegresMod(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float)
    yHat = xMat*ws
    
    CorrCoef = corrcoef(yHat.T, yMat)
    
    print("correlation coefficient is %r" %CorrCoef[0][1])


# 局部加权线性回归函数LWLR, 返回预测值（参数矩阵每次都变，无需返回）
# 为了解决欠拟合问题（偏差方差权衡）
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  # 高斯核对应权重
    
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    y_test_hat = testPoint * ws
    
    return y_test_hat

# 对全体样本做LWLR
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    
    return yHat


# 绘制数据散点图和LWLR的拟合曲线
def LWLRPlot(xArr, yArr, k=1.0):
    import matplotlib.pyplot as plt
    
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],\
               yMat[:, 0].flatten().A[0],\
               s = 2, c = "red")
    
    yHat = lwlrTest(xArr, xArr, yArr, k)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    ax.plot(xSort[:,1], yHat[srtInd])
    
    plt.show()

# 预测值与实际值的平方误差和
def rssError(yArr, yHatArr):
    yArr = array(yArr).astype("float64")
    yHatArr = array(yHatArr).astype("float64")
    return ((yArr - yHatArr)**2).sum()


# 若特征比样本点还多，或其他原因无法求xTx的逆
# 岭回归函数
def ridgeRegres(xArr, yArr, lam=0.2):
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    # 标准化?
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    # 特征矩阵标准化
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)    # 方差
    xMat = (xMat - xMeans)/xVar
    
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom)==0.0:
        print("This matrix is singular, cannot do inverse")
        return
        
    ws = denom.I*(xMat.T*yMat)
    
    return ws

# 岭回归测试, 变化lamda观察回归系数
# plt = 0 时，不绘图
def ridgeTest(xArr, yArr, plt= 1):
    import matplotlib.pyplot as plt
    
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xArr, yArr, exp(i-10))
        wMat[i, :]=ws.T
        
    # 图示 y:回归系数 x:log(lambda)
    if plt != 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(wMat)
        plt.show()
    
    return wMat


# 向前逐步线性回归 eps:步长， numIt：迭代次数
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    #xMat = regularize(xMat)
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    
    for i in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = zeros((numVal, 30))
    # 将数据集随机划分为训练集：测试集=9：1
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        
        random.shuffle(indexList)  # 将序列的所有元素随机排序
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
    
        wMat = ridgeTest(trainX, trainY, 0)
    
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX-meanTrain)/varTrain
            trainY = array(trainY).astype("float64")
            yEst = matTestX*mat(wMat[k,:]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    
    xMat = mat(xArr)
    yMat = mat(yArr, dtype = float).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights/varX
    
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1*sum(multiply(meanX, unReg))+mean(yMat))