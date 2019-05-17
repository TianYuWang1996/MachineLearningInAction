
# 使用类的方式构建CART树的节点
class treeNode():
    def __init__(self, feat, val, left, right):
        self.featureToSpiltOn = feat
        self.valueOfSpilt = val
        self.leftBranch = left
        self.rightBranch = right