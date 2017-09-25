#-*- coding: UTF-8 -*-
import pandas as pd
import random
import operator
import time
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from math import *

#求最多类
def Majority(classlist):
    label_count = {}
    for label in classlist:
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
        sorted_label_count = sorted(label_count.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sorted_label_count[0][0]

#随机森林主函数
def RandomForest(splitSize,sizeOfForest):
    #判断划分数据集比例
    if splitSize > 1.0 or splitSize<0.0:
        return

    preArr=[]#预测结果

    for i in xrange(sizeOfForest):
        print("第"+repr(i+1)+'棵树所选特征为：')
        dataSet,featureSelection,dataY,selectFeaNum = LoadFile()

        print featureSelection
        print("-"*40)
        #划分数据集
        splitNum = int(splitSize*len(dataSet))#
        trainSet = dataSet[:splitNum]
        trainSetY =dataY[:splitNum]
        testSet = dataSet[splitNum:]
        testSetY =dataY[splitNum:]
        #决策树
        clf = tree.DecisionTreeClassifier(criterion='gini')
        clf.fit(trainSet, trainSetY)
        predAnswer = clf.predict(testSet)
        preArr.append(predAnswer)

    transPreArr = map(list, zip(*preArr))#转置

    voteAnswer = []

    #投票求得每棵树预测集合的最大类
    for i in xrange(len(transPreArr)):
        voteAnswer.append(Majority(transPreArr[i]))

    trueNum = 0
    for j in xrange(len(voteAnswer)):
        if voteAnswer[j] == testSetY[j]:
            trueNum += 1
    #算精度
    acc = 1.0 * trueNum / len(voteAnswer)
    prfRes = metrics.classification_report(testSetY,voteAnswer)
    return acc,selectFeaNum,prfRes

#加载数据集，并选择特征
def LoadFile():
    df = pd.read_csv('waveform-+noise.csv')
    dataset = df.values[:,:-1].tolist()
    dataY = df.values[:,-1].tolist()
    features = df.columns.values[0:-1].tolist()
    #T = int(log(len(features), 2))#选择T个特征
    T = int(sqrt(len(features)))
    #T= int(len(features)/2)
    #T =20
    selectFeaNum = T
    subsetIndex = []
    while(T):
        index = random.randint(0,len(features)-1)
        if index not in subsetIndex:
            subsetIndex.append(index)
            T -= 1
    subsetIndex.sort()
    featuresSelection = []#选择的特征名
    for i in subsetIndex:
        featuresSelection.append(features[i])
    n = len(features)
    while(n):
        if features[n-1] not in featuresSelection:
            for q in range(len(dataset)):
                del(dataset[q][n-1])
        n -= 1
    return dataset,featuresSelection,dataY,selectFeaNum

if __name__ == '__main__':
    startTime = time.clock()
    treeNum = 300
    acc,selectFeaNum,prf= RandomForest(0.5, treeNum)
    endTime = time.clock()
    print 'the number of trees : %d' % treeNum
    print 'the number of selected features : %d' % selectFeaNum
    print '*'*50
    print prf
    print '*'*50
    print 'accuracy is %f' % acc
    print 'the run time : %f' % (endTime - startTime)

    print '*'*50
    print '直接调用sklearn中的随机森林学习包：'
    startTime = time.clock()
    df = pd.read_csv('waveform-+noise.csv')
    dataset = df.values[:,:-1].tolist()
    dataY = df.values[:,-1].tolist()
    x_train, x_test, y_train, y_test = train_test_split(dataset, dataY, test_size = 0.5)
    clf = RandomForestClassifier(n_estimators=treeNum,max_features='sqrt')
    clf.fit(x_train,y_train)
    preRes = clf.predict(x_test)
    accSL = metrics.accuracy_score(y_test,preRes)#精度
    prfSL = metrics.classification_report(y_test,preRes)#P,R,F值
    print prfSL
    print '*'*50
    print 'accuracy is %f' % accSL
    endTime = time.clock()
    print 'the run time : %f' % (endTime - startTime)