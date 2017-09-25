#-*- coding: UTF-8 -*-
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
import time
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import tree

#训练次级学习器，并用总测试集的初级样本做预测
def SecondaryLearningModel(xtrain,ytrain,xtest,ytest):

    preArr = []
    clf = linear_model.LinearRegression()
    clf.fit(xtrain,ytrain)
    for i in xrange(len(xtest)):
        preAnswer =float(clf.predict(xtest[i]))
        preArr.append(preAnswer)
    #返回mae
    mae = metrics.mean_absolute_error(ytest,preArr)
    return mae

#预测总测试集，返回测试集的初级样本，用于预测次级学习器
def predictTestData(MLmodel,testData):
    resArr = []
    for i in xrange(len(testData)):
        temp=[]
        for clf in MLmodel:
            answer = float(clf.predict(testData[i]))
            temp.append(answer)
        resArr.append(temp)
    return resArr

#训练初级学习器，预测训练集，返回训练集初级样本，初级学习器模型
def PrimaryLearningModel(dataset,datay,m):
    resArr = []
    PLModel=[]
    trainX,testX,trainY,testY = train_test_split(dataset,datay,test_size=0.5)
    print'开始训练初级学习器！'
    #初级学习器1--KNN
    clf1 = neighbors.KNeighborsClassifier(n_neighbors = 3)
    clf1.fit(trainX,trainY)
    PLModel.append(clf1)

    #初级学习器2--CART
    clf2 = tree.DecisionTreeClassifier(criterion='gini')
    clf2.fit(trainX,trainY)
    PLModel.append(clf2)

    #初级学习器3--线性回归
    clf3 = linear_model.LinearRegression()
    clf3.fit(trainX,trainY)
    PLModel.append(clf3)
    print'初级学习器已经完成！'
    print'开始生成初级样本！'
    for i in xrange(m):
        temp = []
        for item in PLModel:
            preAnswer = float(item.predict(testX[i]))
            temp.append(preAnswer)
        temp.append(testY[i])
        resArr.append(temp)
    print'样本已经生成！'
    return resArr,PLModel

#加载数据集
def LoadFile():
    df = pd.read_csv('OnlineNewsPopularitytrain.csv')
    data = df.values[:,1:-1].tolist()
    datay = df.values[:,-1].tolist()
    return data,datay

if __name__ == '__main__':
    startTime = time.clock()
    #迭代次数
    iteration = 10
    sumRes = 0.0
    for n in xrange(iteration):
        print'第%d次迭代开始：' %(n+1)
        dataSet,dataLabels = LoadFile();
        m = 5000
        #分割总数据集
        trainDate,testData,trainY,testY = train_test_split(dataSet,dataLabels,test_size=0.3)
        #训练初级学习器，并预测生产初级样本
        resData,model= PrimaryLearningModel(trainDate,trainY,m)
        #用初级学习器预测总测试样本，返回测试样本的初级样本
        testPredictRes = predictTestData(model,testData)
        resDataLabels= [i[-1] for i in resData]
        for p in range(len(resData)):
            del(resData[p][-1])
        print'开始训练次级学习器！'
        mae = SecondaryLearningModel(resData,resDataLabels,testPredictRes,testY)

        print'第%d迭代的MAE为: %f ' %(n+1,mae)
        print '*'*30
        sumRes += mae
    endTime = time.clock()
    #多次迭代后MAE的均值
    print 'the aveage of MAE : %f' % (sumRes/iteration)
    print 'the run time : %f' % (endTime - startTime)

