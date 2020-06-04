# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-21 15:27:10
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 13:16:23
# @License: MIT LICENSE

# import csv
import numpy as np
import pandas as pd
import time
import random
import datetime
# from Sensor_Contribution import process
# from datetime import datetime

n = 500
#第一种随时时间方法的起止时间
start1 = '2020-04-20 00:00:00'
end1 = '2020-04-22 00:00:00'
#第二种随时时间方法的起止时间
start2 = '2020-04-20'
end2 = '2020-04-22'
lenth = 8*n

#按序生成的工艺流程
# np.random.seed(1)
# m1 = np.random.randint(1,6,lenth)
# a1 = [i for i in m1]
a1 = [1]*800+[2]*800+[3]*800+[4]*800+[5]*800

#定义工艺流程函数（当前触发的传感器范围）
def process(x,y):
    b6=[0]*(y-x)
    if x < y and y <= 8*n:
        for i in range(x,y):
            b6[i] = random.uniform(0,1)
    return b6

#生成随机的传感类型
# np.random.seed(1)
m1 = np.random.randint(65,70,lenth)
a2 = [chr(m) for m in m1]

#生成随机发送时间

def strTimeProp(start, end, prop, frmt):
    stime = time.mktime(time.strptime(start, frmt))
    etime = time.mktime(time.strptime(end, frmt))
    ptime = stime + prop * (etime - stime)
    return int(ptime)

# def randomTimestamp(start, end, frmt='%Y-%m-%d %H:%M:%S'):
#     return strTimeProp(start, end, random.random(), frmt)

def randomDate(start, end, frmt='%Y-%m-%d %H:%M:%S'):
    return time.strftime(frmt, time.localtime(strTimeProp(start, end, random.random(), frmt)))

def randomDateList(start, end, n, frmt='%Y-%m-%d %H:%M:%S'):
    return [randomDate(start, end, frmt) for _ in range(n)]

#或者使用以下函数产生随机发送时间
def randomtimes(start, end, n, frmt="%Y-%m-%d"):
    stime = datetime.datetime.strptime(start, frmt)
    etime = datetime.datetime.strptime(end, frmt)
    return [random.random() * (etime - stime) + stime for _ in range(n)]

#根据工艺流程设置相应的传感器位置
# np.random.seed(1)
# a4 = np.zeros(shape=(lenth//5,2),dtype=np.int)
a4 = list()
for x in range(0,lenth):
  m1 = np.random.randint(20*(a1[x]-1),20*a1[x],lenth//5)
  m2 = np.random.randint(0,20,lenth//5)
  i = np.random.randint(0,lenth//5)
  a4.append((m1[i],m2[i]))
#   a4 = [(a4x[i],a4y[i]) for i in range(lenth//5)]


# #四列的内容（列长度必须一致）
# a1 = [1]*n+[2]*n+[3]*n+[4]*n+[5]*n+[6]*n+[7]*n+[8]*n
# a2 = [x for x in range(0,lenth)]
# a3 = randomDateList(start1, end1, lenth)
a3 = [0]*8*n
# a3 = randomtimes(start2, end2, lenth)
# a4 = [x for x in range(0,lenth)]
a5 = [x for x in range(0,lenth)]
a6 = process(0,8*n)

dataframe = pd.DataFrame({"technological process":a1,
                          "sensor type":a2,
                          "transmission time":a3,
                          "location":a4,
                          "transmission probability":a6})
                          # "sensor index":a5,

#保存csv文件
dataframe.to_csv(r"data.csv")

# #关闭文件
# f.close()