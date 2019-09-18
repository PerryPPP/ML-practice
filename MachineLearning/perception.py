import numpy as np

# create a dataset which contains 3 samples with 2 classes
def createDataSet():
    # +:[3, 3], [4, 3] -:[1, 1]
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = [1, 1, -1]
    return X, Y

# classify using perception
def perceptionClassify(trainGroup, trainLabels):
    global w, b
    isFind = False  # the flag of find the best w and b
    numSamples = trainGroup.shape[0]
    mLength = trainGroup.shape[1]
    w = [0]* mLength
    b = 0
    while(not isFind):
        for i in np.xrange(numSamples):
            if cal(trainGroup[i],trainLabels[i]) <= 0:
                print("w: " + w + " b: " + b)
                update(trainGroup[i],trainLabels[i])
                break    #end for loop
            elif i == numSamples-1:
                print("w: " + w + " b: " + b)
                isFind = True   #end while loop


def cal(row,trainLabel):
    global w, b
    res = 0
    for i in np.xrange(len(row)):
        res += row[i] * w[i]
    res += b
    res *= trainLabel
    return res
def update(row,trainLabel):
    global w, b
    for i in np.xrange(len(row)):
        w[i] += trainLabel * row[i]
    b += trainLabel