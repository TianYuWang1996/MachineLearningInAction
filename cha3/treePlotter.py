import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 绘制节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",\
                            xytext=centerPt, textcoords="axes fraction",\
                            va="center", ha="center", bbox=nodeType,\
                            arrowprops=arrow_args)

# 绘制节点之间连线的注释文本
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

# 绘制树形图
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # firstStr第一个字典的键，即父节点上的文本
    firstStr = list(myTree.keys())[0]
    # cntrPt当前节点的坐标
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    
    secondDict = myTree[firstStr] # 按键取值，获取第二个字典
    # 更新纵坐标yOff
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    # 遍历字典中的键值
    # 若值的类型为字典，则该分支有后续，递归调用原函数
    # 若值的类型不为字典，则该分支无后续，为叶节点，画出节点及连线注释
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))   # 树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 树的深度
    plotTree.xOff = -0.5/plotTree.totalW           # 初始横坐标
    plotTree.yOff = 1.0                            # 初始纵坐标
    plotTree(inTree, (0.5, 1.0), "")
    plt.show()


# 获取树的叶节点数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs +=1
    return numLeafs


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 示例数据：决策树列表
def retrieveTree(i):
    listOfTrees = [{"no surfacing": {0: "no", 1: {"flippers":\
                   {0: "no", 1: "yes"}}}},
                   {"no surfacing": {0: "no", 1: {"flippers":\
                   {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}
                   ]
    return listOfTrees[i]
