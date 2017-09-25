#-*- coding: UTF-8 -*-
import math
import random
import pandas as pd
import types
import time

#算欧式距离
def distance(x, y):
    '''
    x,y:为两个点
    :return:两点的欧式距离
    '''
    return math.sqrt(sum((a - b) ** 2 for (a, b) in zip(x, y)))

#类内距离
def classInnerDis(tempdata):
    '''
    :param tempdata:一个簇(类)的所有点集合
    :return: 这个簇(类)的任意两点间的欧式距离和
    '''
    lenData = len(tempdata)
    sum=0.0
    for i in xrange(lenData):
        for j in xrange(i,lenData):
            sum+=distance(tempdata[i],tempdata[j])
    return sum

#划分点
def sort(dataset,sortArr,n):
    '''
    :param dataset:所有样本
    :param sortArr: 聚类的簇的集合
    :param n: 划分为n个簇
    :return: 划分完了的簇集合
    '''
    newsortArr = [[] for _ in sortArr]
    for line in dataset:
        disArr = []
        for i in range(len(sortArr)):
            disArr.append(distance(line,sortArr[i]))
        minIndex = disArr.index(min(disArr))
        newsortArr[minIndex].append(line)
    return newsortArr

#求中点
def pointCenter(data):
    '''
    :param data:一个簇(类)
    :return: 返回这个簇的中点
    '''
    centerLoc =[0.0]*len(data[0])
    if len(data) == 1:
        return data
    else:
        for i in range(len(data[0])):
            sum = 0.0
            for j in range(len(data)):
                sum += data[j][i]
            centerLoc[i]=sum/len(data)
        return centerLoc

#随机样本
def randomArray(data,k):
    '''
    :param data:所有样本
    :param k: 去k个随机样本
    :return: 不相同的k个随机样本集合
    '''
    randomArr = random.sample(data,k)
    for i in xrange(k):
        for j in xrange(i+1,k):
            if randomArr[i] == randomArr[j]:
                randomArr =randomArray(data,k)
                break
    return randomArr

#算法函数
def kmeans(dataset,randomArr,n):
    '''
    :param dataset:所有样本集合
    :param randomArr: 随机样本
    :param n: 划分为n个簇
    :return: 划分完以后的簇结合
    '''
    newsortArr = sort(dataset,randomArr,n)
    newRandomArr = []
    if len(newRandomArr) == 0:
        newRandomArr = randomArr
    else:
        for i in range(len(newsortArr)):
            randomArr[i] = pointCenter(newsortArr[i])
        while randomArr != newRandomArr:
            newRandomArr = randomArr
            newsortArr = sort(dataset,newRandomArr,n)
    return newsortArr



if __name__ == "__main__":
    startTime = time.clock()
    df=pd.read_csv('F:\\pyhonProject\\turkiye-student-evaluation_generic.csv')
    data = df.values[:2000,2:].tolist()#过滤前两列
    for i in range(len(data)):
        for j in range(len(data[0])):
            if type(data[i][j]) is types.LongType:
                data[i][j] = float(data[i][j])
    #k取9
    k = 9
    randomArr = randomArray(data,k)
    res= kmeans(data,randomArr,k)
    s=0.0
    for i in range(len(res)):
        s+=classInnerDis(res[i])
    sres=classInnerDis(data)
    endTime = time.clock()
    #输出
    print '属性：%d' %len(data[0])
    print '样本量： %d' %len(data)
    print 'K = ' + str(k)+' 类内距离：' + str(s)
    print '未聚类时的点距离和：%f' %sres
    print '比值是： %f' %(s/sres)
    print '运行时间： %f' %(endTime - startTime)