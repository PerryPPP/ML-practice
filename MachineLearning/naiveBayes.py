import pandas as pd
import numpy as np

def createDataSet():
    # +:[3, 3], [4, 3] -:[1, 1]
    X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
    Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    return X, Y

# 极大似然估计  朴素贝叶斯算法

def naiveBayes(traindata, labels, features):

    #求labels中每个label的先验概率,P(Y = ck)
    labelset = set(labels)
    P_y = {}
    for label in labelset:
        P_y[label] = labels.count(label)/len(labels)

    #求P(x^(j) = ajl | Y = ck)
    P_xy = {}
    for y in P_y.keys():
        y_index = [i for i, label in enumerate(labels) if label == y]  # labels中出现y值的所有数值的下标索引
        for j in range(len(features)):      # features[0] 在trainData[:,0]中出现的值的所有下标索引
            x_index = [i for i, feature in enumerate(trainData[:,j]) if feature == features[j]]
            xy_count = len(set(x_index) & set(y_index))   # set(x_index)&set(y_index)列出两个表相同的元素
            pkey = str(features[j]) + '*' + str(y)
            P_xy[pkey] = xy_count / float(len(labels))
            print(pkey,P_xy[pkey])

    #求条件概率
    P = {}
    for y in P_y.keys():
        for x in features:
            pkey = str(x) + '|' + str(y)
            P[pkey] = P_xy[str(x)+'*'+str(y)] / float(P_y[y])    #P[X1/Y] = P[X1Y]/P[Y]
            print(pkey,P[pkey])

    #求[2,'S']所属类别
    F = {}   #[2,'S']属于各个类别的概率
    for y in P_y:
        F[y] = P_y[y]
        for x in features:
            F[y] = F[y]*P[str(x)+'|'+str(y)]     #P[y/X] = P[X/y]*P[y]/P[X]，分母相等，比较分子即可，所以有F=P[X/y]*P[y]=P[x1/Y]*P[x2/Y]*P[y]
            print(str(x),str(y),F[y])

    features_label = max(F, key=F.get)  #概率最大值对应的类别
    return features_label



