#-*- coding: UTF-8 -*-
import time
from math import log
import operator
import numpy as np
import copy
import PYplot

#字符数组转换为INT数组
def StrToInt(arr):
    for index ,item in enumerate(arr):
        arr[index] = float(item)
    return arr

#连续型数据二值法
def BiPartition(arr):
    midNum = np.median(arr)
    for index, item in enumerate(arr):
        if item <= midNum:
            arr[index] = 0
        else: arr[index] = 1
    return arr

#月份预处理
def EncaseMonth(arr):
    firstQuarter = ["jan","feb","mar"]
    secondQuarter = ["apr","may","jun"]
    thirdQuarter = ["jul","aug","sep"]
    fouthQuarter = ["oct","nov","dec"]
    for index, item in enumerate(arr):
        if item in firstQuarter:
            arr[index] = '1'
        elif item in secondQuarter:
            arr[index] = '2'
        elif item in thirdQuarter:
            arr[index] = '3'
        elif item in fouthQuarter:
            arr[index] = '4'
        else:arr[index] = '0'
    return arr

#预处理
def Pertreatment(url):
    with open(url,'r') as File:
        dataset = File.readlines()
        #获得原始列表列数
        for ele in dataset:
            ele = ele.split(',')
            colNums = len(ele)
            break
        #获得原始列表行数
        rowsNum = len(dataset)

        matrix = []
        matrixY = []
        lenArr = len(dataset) - 1
        #初始化二维数组
        for row in range(lenArr):
            matrix.append([])
            for col in range(colNums ):
                matrix[row].append([])
        #给二维数组赋值（除去非数值的第一行），把最后一列Y值赋值给matrixY
        for rows, ele in enumerate(dataset):
            if rows == lenArr:
                break
            ele = dataset[rows+1].split(',')
            for col,values in enumerate(ele):
                matrix[rows][col] = ele[col]
                if col == colNums - 1:
                    matrix[rows][col] = ele[col] =ele[col].strip('\n').strip('""')
                    break
        ageArr = []
        balanceArr = []
        dayArr = []
        monthArr = []
        durationArr = []
        campaignArr = []
        pdaysArr = []
        previousArr = []
        for ele in matrix:
            ageArr.append(ele[0])
            balanceArr.append(ele[5])
            dayArr.append(ele[9])
            monthArr.append(ele[10].strip('""'))
            durationArr.append(ele[11])
            campaignArr.append(ele[12])
            pdaysArr.append(ele[13])
            previousArr.append(ele[14])

        ageArr = StrToInt(ageArr)
        balanceArr = StrToInt(balanceArr)
        dayArr = StrToInt(dayArr)
        durationArr = StrToInt(durationArr)
        campaignArr = StrToInt(campaignArr)
        pdaysArr = StrToInt(pdaysArr)
        previousArr = StrToInt(previousArr)


        ageArrBP = BiPartition(ageArr)
        balanceArrBP = BiPartition(balanceArr)
        dayArrBP = BiPartition(dayArr)
        monthArrEncase = EncaseMonth(monthArr)
        durationArrBP = BiPartition(durationArr)
        campaignArrBP = BiPartition(campaignArr)
        pdaysArrBP = BiPartition(pdaysArr)
        previousArrBP = BiPartition(previousArr)
        for i in range(len(ageArr)):
            matrix[i][0] = ageArrBP[i]
            matrix[i][5] = balanceArrBP[i]
            matrix[i][9] = dayArrBP[i]
            matrix[i][10] = monthArrEncase[i]
            matrix[i][11] = durationArrBP[i]
            matrix[i][12] = campaignArrBP[i]
            matrix[i][13] = pdaysArrBP[i]
            matrix[i][14] = previousArrBP[i]
        return matrix

#加载数据集
def loadDataSet():
    global labels
    labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
              'duration', 'campaign', 'pdays', 'previous','poutcome','y']
    training_set = Pertreatment('F:\\pyhonProject\\bank-fulltrain.csv')
    test_set = Pertreatment('F:\\pyhonProject\\bank-fulltest.csv')

    return training_set, test_set, labels

#获取label值
def Label(dataSet):
    global currentLabel
    for line in dataSet:
        currentLabel = line[-1]
        if currentLabel not in labelList.keys():
            labelList[currentLabel] = 0
        labelList[currentLabel] += 1
    return labelList

#信息熵
def calEntropy(dataSet):
    global labelList
    numEntries = len(dataSet)
    labelList = {}
    labelList = Label(dataSet)
    entropy = 0.0
    for key in labelList:
        prob = float(labelList[key]) / numEntries
        if prob != 0:
            entropy -= prob * log(prob, 2)  # log base 2
    return entropy

#对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的特征分裂
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calEntropy(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 循环所有的特征
        featList = [example[i] for example in dataSet]  # 穿件一个特征列表
        uniqueVals = set(featList)  # 获得唯一值的set集合
        # if len(uniqueVals) > 10:
        #     featList = biDataSet(i, featList)
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calEntropy(subDataSet)
            splitInfo -= (prob * log(prob, 2))
        # print splitInfo
        if splitInfo != 0:
            infoGainRatio = (baseEntropy - newEntropy) / splitInfo
        else:
            infoGainRatio = 0
        if (infoGainRatio > bestInfoGainRatio):  # 获得最大信息增益率
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature, bestInfoGainRatio

#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#预剪枝函数
def testing(myTree,datatest,labels):
    error=0.0
    for i in range(len(datatest)):
        if classify(myTree,labels,datatest[i])!=datatest[i][-1]:
            error += 1
    #print 'myTree %d' %error
    return float(error)

#目前最多
def testingMajor(major,datatest):
    error=0.0
    for i in range(len(datatest)):
        if major!=datatest[i][-1]:
            error+=1
    #print 'major %d' %error
    return float(error)

#生成树 递归生成决策树
def CreateTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    templabels = copy.deepcopy(labels)
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 所有的类别一样停止划分
    if len(dataSet[0]) == 1:  # 没有特征可以划分
        return majorityCount(classList)
    bestFeat, bestInfoGainRatio = chooseBestFeatureToSplit(dataSet)#返回属性列ID和信息增益比
    if bestInfoGainRatio<0.001:   # 阈值
        return majorityCount(classList)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = CreateTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #预剪枝：进行测试，若划分没有提高准确率，则不进行划分，返回该节点的投票值作为节点类别
    if testing(myTree,dataSet,templabels)<testingMajor(majorityCount(classList),dataSet):
        return myTree
    #print (testing(myTree,dataSet,templabels),testingMajor(majorityCnt(classList),dataSet))
    return myTree

#遍历树
def classify(inputTree, featLabels, testVec):
    global  classLabelres
    firstStr = inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabelres=classify(secondDict[key],featLabels,testVec)
            else:
                classLabelres = secondDict[key]
    return classLabelres

#精确度
def Classify(tree, testset):
    labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
              'duration', 'campaign', 'pdays', 'previous','poutcome','y']
    result = []
    #计算TP,FP,FN,TN
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for ins in testset:
        tmp = classify(tree, labels, ins)
        result.append(tmp == ins[-1])
        if tmp == ins[-1]:
            if ins[-1] == 'yes':
                TP += 1
            else: TN += 1
        else:
            if ins[-1] == 'yes':
                FN += 1
            else: FP += 1
    accuracy = float(result.count(True)) / float(len(result))
    return accuracy,TP,FP,FN,TN

if __name__ == '__main__':
    classLabelres = 0
    starttime = time.time()
    trainingset, testset, labels = loadDataSet()
    tree = CreateTree(trainingset, labels)
    #计算accuracy，TP,FP,FN,TN
    accuracy,tp,fp,fn,tn = Classify(tree,testset)
    print ' Result : '
    print '------------------------------------'
    print ' 属性 ： '+ str(len(trainingset[0])-1)
    print ' 训练集样本量 ： ' + str(len(trainingset))
    print ' 测试集样本量 ： ' + str(len(testset))
    print '------------------------------------'
    print " accuracy: %.4f" % accuracy
    print '------------------------------------'
    print ' |'+' 真正例(TP) : '+str(tp)+' | '+' 假反例(FN) : ' + str(fn) + ' | '
    print ' |'+' 假正例(FP) : '+str(fp)+' | '+' 真反例(FN) : ' + str(tn) + ' | '

    dataLen = len(testset)
    yesPrecision = float(tp) / (tp + fp)
    yesRecall = float(tp) / (tp + fn)
    yesF1 = float(2 * tp) / (dataLen + tp - tn)
    print '------------------------------------'
    print ' the yes of precision : %4f '% yesPrecision
    print ' the yes of recall : %4f ' % yesRecall
    print ' the yes of F1 ：%4f' % yesF1
    print '------------------------------------'
    noPrecision = float(tn) / (fn + tn)
    noRecall = float(tn) / (tn + fp)
    noF1 = float(2 * tn) / (dataLen + tn -tp)
    print ' the no of precision : %4f' %noPrecision
    print ' the no of recall : %4f' %noRecall
    print ' the no of F1 ：%4f' %noF1
    print '------------------------------------'
    #PYplot.createPlot(tree)
    endtime = time.time()
    print " the run time of program is %4f s" %( endtime - starttime)