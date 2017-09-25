#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from math import *
import random
import time

#加载数据集
def loadFile():
    df=pd.read_csv('F:\\pyhonProject\\FeaturesSelect\\waveform-+noise.csv')
    dataFea = df.values[:,:-1].tolist()
    dataLable = df.values[:,-1].tolist()
    return dataFea,dataLable

#J48分类算法
def J48Classification(dataFea, dataLabel):
    xtrain, xtest, ytrain, ytest = train_test_split(dataFea, dataLabel, test_size=0.2)
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(xtrain, ytrain)
    answer = clf.predict(xtest)
    res = metrics.classification_report(ytest,answer)
    acc = metrics.accuracy_score(ytest,answer)
    return res,acc

#SVM分类算法
def SVMClassification(dataFea, dataLabel):
    xtrain, xtest, ytrain, ytest = train_test_split(dataFea, dataLabel, test_size=0.2)
    svclf = SVC(kernel = 'linear')
    svclf.fit(xtrain,ytrain)
    preAnswer = svclf.predict(xtest)
    res = metrics.classification_report(ytest,preAnswer)
    acc = metrics.accuracy_score(ytest,preAnswer)
    return res,acc

#LIBASED特征选择
def L1basedFeaSele(dataFea, dataLabel):
    newData= LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(dataFea, dataLabel)
    return newData

#PCA降维
def PCADimension(dataFea,k):
    pca = PCA(n_components=k)
    newData = pca.fit_transform(dataFea)
    return newData

#乘积之和
def multipl(a,b):
    sumofab=0.0
    for i in range(len(a)):
        temp=a[i]*b[i]
        sumofab+=temp
    return sumofab

#皮尔逊系数
def PearsonCorrelation(x,y):
    n=len(x)
    #求和
    sum1=sum(x)
    sum2=sum(y)
    #求乘积之和
    sumofxy=multipl(x,y)
    #求平方和
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
    den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
    return abs(num/den)

#特征扩展迭代
def FeaExtendIter(data,k,feaLabel):

    res,accFE1 = J48Classification(data,dataLabel)
    lenData = len(data[0])
    randomIndex = random.randint(0,lenData-1)
    randomArr = [x[randomIndex] for x in data]
    #print randomArr
    pearsonArr=[]
    data=np.delete(data,randomIndex,axis=1)
    #print len(dataFea[0])
    for i in xrange(len(data[0])):
        Y = [x[i] for x in data]
        pearsonRes = PearsonCorrelation(randomArr,Y)
        pearsonArr.append(pearsonRes)
    minIndex = pearsonArr.index(min(pearsonArr))
    #print(maxIndex,pearsonArr[maxIndex])
    newArr=[]
    colValue = [x[minIndex] for x in data]
    for j in xrange(len(randomArr)):
        newArr.append(randomArr[j] + colValue[j])
    #print randomArr

    data=np.column_stack((data,randomArr))
    data=np.column_stack((data,newArr))
    k += 1
    res,accFE2=J48Classification(data,dataLabel)
    if accFE1 > accFE2:
        data=np.delete(data,-1,axis=1)
    return data,k

#特征扩展
def FeaExtend(dataFea,feaLabel):
    k = 0
    newData,k =FeaExtendIter(dataFea,k,feaLabel)
    while(k < 60):
        newData,k = FeaExtendIter(newData,k,feaLabel)
    return newData

if __name__ == '__main__':
    startTime = time.clock()
    dataFea,dataLabel = loadFile()

    print '---------------------原数据------------------------'
    print '---------------------J48算法-----------------------'
    metaDJ48StartTime = time.clock()
    res1,acc1 = J48Classification(dataFea,dataLabel)
    print res1
    print 'accuracy：%f' % acc1
    metaDJ48EndTime = time.clock()
    print '运行时间: %f' % (metaDJ48EndTime - metaDJ48StartTime)

    print '---------------------SVM算法------------------------'
    metaDSVMStartTime = time.clock()
    res2,acc2 = SVMClassification(dataFea,dataLabel)
    print res2
    print 'accuracy：%f' % acc2
    metaDSVMEndTime = time.clock()
    print '运行时间: %f' % (metaDSVMEndTime - metaDSVMStartTime)


    print '-------------------特征转换(PCA)--------------------'
    print '---------------------J48算法------------------------'
    newPCAdata = PCADimension(dataFea, 20)
    pcaDJ48StartTime = time.clock()
    res3,acc3 = J48Classification(newPCAdata, dataLabel)
    print res3
    print 'accuracy：%f' % acc3
    pcaDJ48EndTime = time.clock()
    print '运行时间: %f' % (pcaDJ48EndTime - pcaDJ48StartTime)

    print '----------------------SVM算法-----------------------'
    pcaDSVMStartTime = time.clock()
    res4,acc4 = SVMClassification(newPCAdata, dataLabel)
    print res4
    print 'accuracy：%f' % acc4
    pcaDSVMEndtime = time.clock()
    print '运行时间: %f' % (pcaDSVMEndtime - pcaDSVMStartTime)


    print '-----------------------特征选择---------------------'
    print '-----------------------J48算法----------------------'
    newLBFS = L1basedFeaSele(dataFea, dataLabel)
    lbDJ48StartTime = time.clock()
    res5,acc5 = J48Classification(newLBFS, dataLabel)
    print res5
    print 'accuracy：%f' % acc5
    lbDJ48EndTime = time.clock()
    print '运行时间: %f' % (lbDJ48EndTime - lbDJ48StartTime)

    print '------------------------SVM算法----------------------'
    lbDSVMStartTime = time.clock()
    res6,acc6 = SVMClassification(newLBFS, dataLabel)
    print res6
    print 'accuracy：%f' % acc6
    lbDSVMEndTime = time.clock()
    print '运行时间: %f' % (lbDSVMEndTime - lbDSVMStartTime)


    print '-------------------------特征扩展--------------------'
    print '-------------------------J48算法---------------------'
    newDataFeaExtend = FeaExtend(dataFea,dataLabel)
    feDJ48StartTime = time.clock()
    res7,acc7 = J48Classification(newDataFeaExtend, dataLabel)
    print res7
    print 'accuracy：%f' % acc7
    feDJ48EndTime = time.clock()
    print '运行时间: %f' % (feDJ48EndTime - feDJ48StartTime)

    print '-------------------------SVM算法-------------------'
    feDSVMStartTime = time.clock()
    res8,acc8= SVMClassification(newDataFeaExtend, dataLabel)
    print res8
    print 'accuracy：%f' % acc8
    feDSVMEndTime = time.clock()
    print '运行时间: %f' % (feDSVMEndTime - feDSVMStartTime)


    endTime = time.clock()
    print '----------------------------------------------------'
    print 'the run time of progress is : %f' % (endTime - startTime)