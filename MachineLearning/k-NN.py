from math import sqrt
import operator as opt
import numpy as np


def createDataSet():
    X = np.array([[2, 3], [3, 1], [6, 8], [7, 7]])
    Y = ['a', 'a', 'b', 'b']
    return X, Y

def normData(dataSet):
    maxVals = dataSet.max(axis=0)
    minVals = dataSet.min(axis=0)
    ranges = maxVals - minVals
    retData = (dataSet - minVals) / ranges
    return retData, ranges, minVals

def kNN(dataSet, labels, testData, k):
    #array相减的时候，维度会统一哒> <
    distArray = ((dataSet - testData) ** 2).sum(axis=1) ** 0.5
    sortedIndices = distArray.argsort() # 排序，得到排序后的下标
    indices = sortedIndices[:k] # 取最小的k个
    labelCount = {} # 存储每个label的出现次数
    for i in indices:
        label = labels[i]
        labelCount[label] = labelCount.get(label, 0) + 1 # 次数加一
    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True) # 对label出现的次数从大到小进行排序
    return sortedCount[0][0] # 返回出现次数最大的label

if __name__ == "__main__":
    x, y = createDataSet()
    normDataSet, ranges, minVals = normData(x)
    testData = np.array([6.6, 7.5])
    normTestData = (testData - minVals) / ranges
    result = kNN(normDataSet, y, normTestData, 2)
    print(result)