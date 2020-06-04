# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-22 13:22:34
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 20:08:59
# @License: MIT LICENSE

import csv
import pandas as pd
import numpy.matlib
import numpy as np
import random
import threading
import time
import datetime
import math
# import Sensor_Genaration


#各种参数
n = 500
# a6 = [0]*8*n
# starttime = '2020-04-20 00:00:00'
# endtime = '2020-04-22 00:00:00'
h = 300  #一次完整运行时间是300 min
TTI = h*4*10/(8*n)  #transmission time interval
r = {} #存放候补节点集
lr = 5.00 #挑选预留节点的距离阈值

#读取1-4列数据，以第4列作为索引
# data = pd.read_csv('data.csv', index_col=3, usecols=[i for i in range(1, 5)])
data = pd.read_csv('data.csv')

#获取传输概率那一列的值
a2 = data[u'sensor type']
a3 = data[u'transmission time']
a4 = data[u'location']
a6 = data[u'transmission probability'].astype('float')

# #定义工艺流程函数
# （当前触发的传感器范围）
# def process(x,y):
#     b6=[0]*(y-x)
#     if x < y and y <= 8*n:
#         for i in range(x,y):
#             b6[i] = random.uniform(0,1)
#     return b6

# data['transmission probability'] = process(9,600)


#将传感器与三个特性对应
def sensor(i):
    if i <= data.index[8*n-1]:
        s = data.loc[i]
    process = s['technological process']
    sen_type = s['sensor type']
    location = s['location']
    probability = s['transmission probability']
    # print(process,sen_type,location)
    return process,sen_type,location,probability

#模拟每个时刻传感器触发
#for T in range(0,4): #timeslot
def T_activate(T):
    # process(1000*T,1000*(1+T))  #每个传感器的触发概率是固定的
    start = TTI*T
    end = TTI*(T+1)
    for i in range(10*T,10*(1+T)): #10个minislot
        for m in range(4*i,4*(i+1)):
#             process1,sen_type,location,probability = sensor(m)
            if a6[m] >= 0.7:
                # a3[m] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))#加上分秒微秒
                # a3[m] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                # print(m,a3[m],a6[m])
                a3.loc[m] = random.uniform(start,end)
                #time.sleep(0.1) #钢板运行速度
                pass
    return a3

a3 = T_activate(0)

#定义sensor选择函数：根据时间、传感类型和位置进行特征节点选举
#T时刻x的序号范围是(40*T，40*(T+1))
def select_sensor(x,T):
    r[x] = set()
    for x in range(40*T,40*(T+1)):
        if a3[x]!=0:
            for i in range(0,8*n):
                if a3[i]!=0 and abs(a3[x] - a3[i]) < 0.5 and a2[x] == a2[i]:
                    r[x].add(i)
                elif math.sqrt((eval(a4[i])[0]-eval(a4[x])[0])**2+(eval(a4[i])[1]-eval(a4[x])[1])**2) <= lr:
                    r[x].add(i)
    return r[x]
# select_sensor(0)

#定义高斯分布概率(均值，方差，x)
def Gaussian(a,b,x):
    f = 1/((2*math.pi())**0.5)*(b**0.5)*math.exp(-1*((x-a)**2)/(2*b))
    return f


#计算T时刻节点y与特征节点的后验概率
# def posterior(y,T):
#     R = select_sensor(y,T)
#     Pys = 0
#     Ry = []
#     Ps = 1
#     if a6[y] >= 0.7 #概率阈值需与T_activate()函数中的概率阈值保持一致
#         Py = a6[y]
#     for m in R[y]:
#         Ry.append(a6[m])
# #计算先验概率均值:
#     mean = np.mean(Ry)
# #计算先验概率P(y|S)：
#     var = np.var(Ry)
#     for i in range(len(Ry)):
#         Pys* = 1-Gaussian(mean,var,Ry[i])
#     Pys = 1-Pys
# #计算P(S):
#     for n in range(len(Ry)):
#         Ps* = 1-Ry[n]
#     Ps = 1-Ps
# #计算后验概率P(S|y)：
#     Psy = Pys*Ps/Py
#     return Psy,Py,Ps,Pys

#计算T时刻节点y与特征节点集的互信息MI：
# def MI(y,T):
#     R = select_sensor(y,T)
#     Psy,Py,Ps,Pys = posterior(y,T)
#     P = 0
#     Ry = []
#     for m in R[y]:
#         Ry.append(a6[m])
# #计算先验概率均值:
#     mean = np.mean(Ry)
# #计算先验概率方差：
#     var = np.var(Ry)
# #计算互信息MI:
#     for i in range(len(Ry)):
#         P = Gaussian(mean,var,Ry[i])*Ry[i]
#         M+ = w*math.log(w/(Ry[i]*Py),2)
#     return M

#计算节点y和特征节点集的卡方检验：
# def X_square(y,T):
#     Psy,Py,Ps,Pys = posterior(y,T)
#     Ry = []
#     for m in R[y]:
#         Ry.append(a6[m])
# #计算先验概率均值:
#     mean = np.mean(Ry)
# #计算先验概率方差：
#     var = np.var(Ry)
# #计算联合概率P(y,s)总和——真实值:
#     for i in range(len(Ry)):
#         Pk+ = Gaussian(mean,var,Ry[i])*Ry[i]
#     Pk* = len(Ry)
# #计算P(y,s)理论值：
#     Pe = len(Ry)*Py*Ps
# #计算卡方检验值：
#     Px = (Pk-Pe)**2/Pe
#     return Px

#计算历史传输成功率：
# def suc_gate(y,T):
#     for t in range(T):

##计算T时刻节点y与特征节点的后验概率
def Correlation(y,T):
    r[y] = select_sensor(y,T)



#计算sensor contribution:
def Sen_Contri(y,T):
    c = max(posterior(y,T),MI(y,T),X_square(y,T))
    return c



# for i in range(1,(len(Ry)+1)):
#     Ps.append(Ry[i-1])
#     Pst = Ps[i-1]*Ps[i]




#将生成的数据写入文件
data['transmission time'] = a3
# data['transmission probability'] = a6
data.to_csv(r'data.csv',mode = 'w',index = False)


# for i in data.index:
#     sensor = data.loc[i]
#     process = sensor['technological process']
#     sen_type = sensor['sensor type']
#     location = sensor['location']
# print(process, sen_type, location)
#
