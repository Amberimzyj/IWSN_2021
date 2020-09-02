# Project: iwsn
# File Created: Friday, 26th June 2020 7:21:59 pm
# Author: Yajing Zhang (amberimzyj@qq.com)
# -----
# Last Modified: Friday, 26th June 2020 7:35:39 pm
# Modified By: Yajing Zhang (amberimzyj@qq.com>)
# -----
# Copyright 2017 - 2020 Your Company, Your Company


import os
import random
import math
from ast import literal_eval
from typing import List, Tuple, Union

from indexedproperty import indexedproperty
import numpy as np
import pandas as pd
# from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from iwsn.utils.patterns import singleton
# from iwsn.utils.naive_bayes import NavieBayes


# @singleton
class SensorContrib(object):
    '''The sensor contribution class.'''

    def __init__(self,
                 data_path: str = 'data/6000.csv',
                 # active_thresh: float = 0.2,
                 sensor_num_pre_t: int = 60,
                 trans_time_interal: int = 3,
                 #  feature_sensor_dis: float = 3.,
                 act_num=40,
                 res_num=20,
                 bayes_type: str = 'MultinomialNB'
                 ):
        """The sensor conttibution class.

        Keyword Arguments:
            data_path {str} -- Path to load dataframe. (default: {'data/sensors.csv'})

        Raises:
            FileNotFoundError: The input data path not found.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'data path "{data_path}" not exists.')

        self._data = pd.read_csv(data_path)
        # self._active_th = active_thresh
        self._sensor_npt = sensor_num_pre_t
        self._n_timeslots = int(len(self._data) / self._sensor_npt)
        self._tti = trans_time_interal
        # self._feat_sensor_dis = feature_sensor_dis

        self._path = data_path
        self._length = len(self._data)
        self._act_num = act_num
        self.res_num = res_num
        # self._naive_bayes = NavieBayes(bayes_type)

    # def static_stage(self, max_sensors: int = 10):
    #     if self._n_timeslots <= 2:
    #         raise ValueError("数据的timeslots必须大于2")

    #     # 获得前两个time slot激活的节点
    #     activated_t0 = self.t_activate(0)
    #     activated_t1 = self.t_activate(1)

    #     # ave_pp_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
    #     # ave_mi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
    #     # ave_chi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')

    #     # 依次遍历后续的time slot
    #     for t in range(2, self._n_timeslots - 1):
    #         # self._naive_bayes = NavieBayes("MultinomialNB")
    #         train_X_index, train_X, train_y = self.nb_gen_train_data(
    #             activated_t0, activated_t1)
    #         self._naive_bayes.fit(train_X_index, train_X,
    #                               train_y, partial=True)

    #         eval_X_index, eval_X, eval_y, eval_length = self.select_sensor(
    #             activated_t1, t)
    #         predict_proba = self._naive_bayes.predict_proba(eval_X)[:, 0]
    #         predict_sensors = np.take(
    #             eval_X_index, np.argsort(predict_proba)[:max_sensors])

    #         activated_t0 = activated_t1
    #         activated_t1 = self.t_activate(t)

    #         acc = len(np.intersect1d(activated_t1, predict_sensors)
    #                   ) / len(activated_t1)
    #         print(f'slot: {t}, acc: {acc}')

        # ave_pp_accs[t-1] = self._nb_cal_succ_predict_radio(
        #     t, max_sensors, data_X, data_y, 'posterior_prob')
        # ave_mi_accs[t-1] = self._nb_cal_succ_predict_radio(
        #     t, max_sensors, data_X, data_y, 'mutual_information')
        # ave_chi_accs[t-1] = self._nb_cal_succ_predict_radio(
        #     t, max_sensors, data_X, data_y, 'chi_square_test')

        # ave_pp_accs = ave_pp_accs.mean(axis=0)
        # ave_mi_accs = ave_mi_accs.mean(axis=0)
        # ave_chi_accs = ave_chi_accs.mean(axis=0)

        # plt.plot(ave_pp_accs, color='green')
        # plt.plot(ave_mi_accs, color='blue')
        # plt.plot(ave_chi_accs, color='red')
        # plt.show()

    # def _nb_cal_succ_predict_radio(self, t: int, max_sensors: int, X: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    #     indexed_metrics = self._naive_bayes.select(X, y, metric)
    #     activated = np.array(self.t_activate(t))
    #     accs = self._cal_succ_predict_radio(indexed_metrics[:, 0], activated)
    #     filled_accs = np.full(max_sensors, accs[-1], 'float')
    #     filled_accs[:len(accs)] = accs[:max_sensors]

    #     return filled_accs

    # def _cal_succ_predict_radio(self, predict, actual):
    #     predict = predict.astype('int')

    #     accs = np.empty(len(predict), 'float')
    #     for i in range(1, len(predict)+1):
    #         inter = np.intersect1d(predict[:i], actual)
    #         acc = len(inter) / len(actual)
    #         accs[i-1] = acc

    #     return accs

    def cal_pre_accu(self, t: int, probability: np.ndarray) -> float:
        """计算预测精度

        Args:
            t (int): 要计算第几个timeslot的预测精度
            probability (np.ndarray): 要计算的概率

        Returns:
            float: 预测精度
            list: 预测节点集
            list：触发节点集
        """

        # cond_pro = np.random.rand(self._length,self._length)
        # act_times, joint_times, pri_pro, cond_pro = sensor_contirb.circul(cir_num)
        activated_t1 = self.t_activate(t)  # 前一时刻触发的节点list
        # sel_sensor = np.zeros((self._length,self._length))
        # sort_sensor = np.zeros((self._length,self._length))
        # for i in range(self._sensor_npt):
        #     sel_sensor[i] = cond_pro[activated_t1[i]]
        sel_sensor = probability[activated_t1]
        sel_sensor = np.cumsum(sel_sensor, axis=0)  # 将每一个已触发节点对应的特征节点的条件概率按行相加
        if len(sel_sensor) > 0:
            sel_sensor = sel_sensor[-1]  # 只取最后一行，记录了预测节点对于已触发节点的条件概率的总和
        elif len(sel_sensor) == 0:
            pass
        # sel_sensor = sel_sensor.ravel()[np.flatnonzero(sel_sensor)]
        for i in range(len(sel_sensor)):
            if sel_sensor[i] != 0:
                sel_sensor[i] += 2
            else:
                pass
        sort_sensor = np.argsort(sel_sensor)[::-1]  # 从大到小排序条件概率p(x|i)对应的x
        # sort_sensor = np.flip(np.argsort(sel_sensor),axis=0)#从大到小排序条件概率p(x|i)对应的x
        # 每一行（y）取概率最大的前res_num个x————预测节点集
        sort_sensor = sort_sensor[0: self.res_num]
        # unique, counts = np.unique(sort_sensor, return_counts=True)
        # sort_sensor = unique[np.argsort(counts)[::-1]] #从大到小排序index
        # sort_sensor = np.argsort(sort_sensor)[::-1] #从大到小排序index
        # sort_sensor = sort_sensor[:self._act_num] #显示act_num个index
        activated_t2 = self.t_activate(t+1)  # 后一时刻触发的节点list
        res_sensor = np.intersect1d(sort_sensor, activated_t2)  # 预留的触发节点集
        pre_accu = len(res_sensor)/self._act_num

        return pre_accu, sort_sensor, res_sensor
        # return pre_accu, sel_sensor,sort_sensor,activated_t1, activated_t2

    def cal_ave_pre_accu(self, t1: int, t2: int, probability: np.ndarray) -> float:

        pre_accu = 0
        for i in range(t1, t2):
            accu, _, _ = self.cal_pre_accu(i, probability)
            pre_accu += accu
        ave_pre_accu = pre_accu/(t2-t1)

        return ave_pre_accu

    def t_activate(self, t: int) -> List[int]:
        """Calculate the activate sensors at timeslot t.

        Arguments:
            t {int} -- the specified timeslot.
            sensor_num (int): 每个timesl允许触发的传感器数量 <= 每个时刻激活的传感器数量

        Returns:
            List[int] -- Return the activated sensors' index.
        """

        activated = []
        # active_th = np.linspace(0.5,1,sensor_num,endpoint=False) #设置传感器触发概率随着sensor index递减
        for i in range(self._sensor_npt * t, (self._sensor_npt * t + self._act_num)):
            if self.trans_prob[i] >= 0.2:
                activated.append(i)

        return activated

    # def require_distance(self, distance: int) -> bool:
    #     """判断两个节点间的距离是否小于阈值

    #     Args:
    #         distance (int): 输入的节点间距离

    #     Returns:
    #         bool: 返回判断结果
    #     """
    #     return distance < self._feat_sensor_dis and distance > 0

    # def select_sensor(self, activated: list, t: int) -> Tuple[list, list, list, int]:
    #     """根据上一时隙激活的节点选择后续的节点

    #     Args:
    #         activated (list): 输入的上一时隙激活节点列表
    #         t (int): 当前时隙time slot

    #     Returns:
    #         Tuple[list, list, list, int]: 返回选择后的节点数据
    #     """
    #     length = 0

    #     data_X_index = []
    #     data_X = []
    #     data_y = []

    #     for y in activated:
    #         for x in range(self._sensor_npt * t, self._length):
    #             distance = self._cal_dis(self.location[x], self.location[y])
    #             if self.require_distance(distance):
    #                 length += 1
    #                 data_X_index.append(x)
    #                 data_X.append([distance])
    #                 data_y.append(y)

    #     return data_X_index, data_X, data_y, length

    # def cal_pro(self, cir_num:int, y: int, x: int, yx_num:int) -> np.ndarray:
    #     """最大似然：估计上一时刻触发节点y和节点x之间的条件概率、节点y的先验概率和xy的联合概率

    #     Args:
    #         cir_num: 总循环次数
    #         y (int): 上一时刻触发的节点
    #         x (int): 要预测的节点
    #         yx_num (int):节点y出现后节点x也出现的次数

    #     Returns:
    #         np.ndarray[y][x]: y触发后x触发的概率
    #     """
    #     return np.ndarray[y][x] = yx_num/cir_num

    def circul(self, cir_num: int) -> np.ndarray:
        """模拟传感器循环触发

        Args:
            cir_num (int): 循环次数
            sensor_num (int): 每个timeslot允许触发的节点数量

        Returns:
            np.ndarray: 节点触发次数、共同触发次数
        """

        act_times = np.zeros((self._length), 'int64')  # 记录每个节点在历次循环中触发的次数
        joint_times = np.zeros((self._length, self._length), 'int64')  # 记录两个节点共同触发的次数

        # 记录各种触发次数
        for i in tqdm(range(cir_num)):
            for t in tqdm(range(self._n_timeslots)):
                activated_t1 = self.t_activate(t)  # t时刻触发的节点
                if t < (self._n_timeslots - 1):
                    activated_t2 = self.t_activate(t+1)  # t+1时刻触发的节点
                    joint_times[np.ix_(activated_t1, activated_t2)] += 1
                act_times[activated_t1] += 1
            self._data["transmission probability"] = np.random.uniform(
                0, 1, self._length)
            # self._data["transmission probability"] = [random.uniform(
            #     0, 1) for i in range(self._length)]  # 每循环一次重新生成触发概率

        return act_times, joint_times

    def cal_cond_pro(self, act_times: np.ndarray, joint_times: np.ndarray, cir_num: int) -> np.ndarray:
        """计算条件概率

        Args:
            act_times (np.ndarray): 触发次数array
            joint_times (np.ndarray): 联合触发次数array
            cir_num (int):循环次数

        Returns:
            np.ndarray: 先验概率，条件概率，先验概率倒数，联合概率
        """

        # 计算各种触发概率
        pri_pro = np.zeros((self._length), 'float')  # 存放先验概率
        pri_pro_re = np.zeros((self._length), 'float')  # 存放先验概率的倒数，便于计算条件概率
        cond_pro = np.zeros((self._length, self._length), 'float')  # 存放条件概率
        joint_pro = np.zeros((self._length, self._length), 'float')  # 存放联合概率
        # for a in tqdm(range(self._length)):
        #     pri_pro[a] = act_times[a]/cir_num
        #     for b in tqdm(range(self._length), leave=False):
        #         if pri_pro[a] == 0:
        #             cond_pro[a,b] = 0
        #         else:
        #             cond_pro[a,b] = joint_times[a,b]/(cir_num * pri_pro[a])  #a=y,b=x,计算p(x|y)
        pri_pro = act_times/cir_num
        for j in range(self._length):
            if pri_pro[j] != 0:
                pri_pro_re[j] = 1/pri_pro[j]
            else:
                pri_pro_re[j] = 0
        # 扩展先验概率倒数至self._length维并转置，方便进行矩阵运算
        pri_pro_re = np.tile(pri_pro_re, (self._length, 1)).T
        # cond_pro = np.dot((joint_times/cir_num),(np.matrix(pri_pro).I)) #计算条件概率
        joint_pro = joint_times/cir_num
        cond_pro = joint_pro * pri_pro_re  # 计算条件概率

        return pri_pro, cond_pro, pri_pro_re, joint_pro

    def cal_MI(self, act_times: np.ndarray, joint_times: np.ndarray, cir_num: int) -> float:
        """计算互信息

        Args:
            act_times (np.ndarray): 节点触发次数
            joint_times (np.ndarray): 节点联合触发次数

        Returns:
            float: 互信息概率
        """
        MI_pro = np.zeros((self._length, self._length), 'float')
        pri_pro, cond_pro, pri_pro_re, joint_pro = self.cal_cond_pro(
            act_times, joint_times, cir_num)
        pxpy = np.expand_dims(pri_pro, axis=1) * \
            np.expand_dims(pri_pro, axis=0)  # 计算p(x)*p(y)
        pxpy = np.where(pxpy > 0, 1/pxpy, 0)
        MI_pro = joint_pro * \
            np.log2(np.where((joint_pro * pxpy) > 0,
                             joint_pro * pxpy, 0))  # 计算互信息
        MI_pro[np.isnan(MI_pro)] = 0  # 将nan替换为0

        return MI_pro
        # return pxpy

    def cal_X2(self, act_times: np.ndarray, joint_times: np.ndarray, cir_num: int) -> float:
        """计算卡方概率

        Args:
            act_times (np.ndarray): 节点触发次数
            joint_times (np.ndarray): 节点联合触发次数

        Returns:
            float: 卡方概率
        """

        pri_pro, cond_pro, pri_pro_re, joint_pro = self.cal_cond_pro(
            act_times, joint_times, cir_num)

        # 计算p(x)*p(y)
        pxpy_re = np.expand_dims(pri_pro, axis=1) * \
            np.expand_dims(pri_pro, axis=0)  # 计算临时p(x)*p(y)
        pxpy_re = np.where(pxpy_re > 0, 1/pxpy_re, 0)
        index = np.array(np.nonzero(cond_pro)).T  # 取出条件概率不为0的index
        pxpy = np.zeros((self._length, self._length),
                        'float')  # 存放整形后的p(x)*p(y)
        for i in range(len(index)):
            pxpy[index[i, 0], index[i, 1]] = np.array(
                [pxpy_re[index[i, 0], index[i, 1]]])

        # X2_pro = np.array([pxpy[index[0,0],index[0,1]]])
        X2_pro = (joint_pro-pxpy) * (joint_pro-pxpy) * pxpy
        # X2_pro[np.isnan(X2_pro)] = 0

        return X2_pro
        # return pxpy, X2_pro

    # def nb_gen_train_data(self, activated_t1: list, activated_t2: list) -> Tuple[list, list, list]:
    #     """生成朴素贝叶斯训练数据

    #     Args:
    #         activated_t1 (list): 前一个时刻触发的节点集
    #         activated_t2 (list): 当前时刻触发的节点集

    #     Returns:
    #         Tuple[list, list, list]: 训练的数据
    #     """
    #     data_X_index = []
    #     data_X = []
    #     data_y = []

    #     start_y = max(activated_t2)
    #     for y in activated_t1:
    #         for x in activated_t2:
    #             distance = self._cal_dis(self.location[x], self.location[y])
    #             data_X_index.append(x)
    #             data_X.append([distance])
    #             data_y.append(1)

    #         # for x in random.sample(range(start_y+1, self._length), len(activated_t2)):
    #         for x in range(start_y+1, self._length):
    #             distance = self._cal_dis(self.location[x], self.location[y])
    #             if self.require_distance(distance):
    #                 data_X_index.append(x)
    #                 data_X.append([distance])
    #                 data_y.append(0)

    #     return data_X_index, data_X, data_y

    def save_data(self):
        '''Save dataframe to csv.'''
        self._data.to_csv(self._path)

    # def _cal_dis(self, x: int, y: int) -> float:
    #     """计算两个节点间的距离

    #     Args:
    #         x (int): 节点x的位置
    #         y (int): 节点y的位置

    #     Raises:
    #         TypeError: 输入的位置类型错误
    #         ValueError: 输入的位置值错误

    #     Returns:
    #         float: 返回计算后的距离
    #     """
    #     for a in (x, y):
    #         if a < 0:
    #             raise ValueError("输入的位置必须大于等于0")

    #     return int((x - y) / 10)

    def plot_pic(self, act_times: int, joint_times: int, cir_num: int):
        """绘图

        Args:
            act_times (int): 节点触发次数
            joint_times (int): 联合触发次数
            cir_num (int): 循环次数
        """
        pro_ave_accs = []
        act_times = np.zeros((self._length), 'int64')
        joint_times = np.zeros((self._length, self._length), 'int64')
        for i in range(1, cir_num):
            act_times_temp, joint_times_temp = self.circul(1)
            act_times += act_times_temp
            joint_times += joint_times_temp
            pri_pro, cond_pro, pri_pro_re, joint_pro = self.cal_cond_pro(
                act_times, joint_times, i)
            MI_pro = self.cal_MI(act_times, joint_times, i)
            X2_pro = self.cal_X2(act_times, joint_times, i)
            cond_ave_accs.append(self.cal_ave_pre_accu(0, 9, cond_pro))
            MI_ave_accs.append(self.cal_ave_pre_accu(0, 9, MI_pro))
            X2_ave_accs.append(self.cal_ave_pre_accu(0, 9, X2_pro))
        plt.plot(np.arange(1, cir_num), cond_ave_accs, 'b', label='CP')
        plt.plot(np.arange(1, cir_num), MI_ave_accs, 'g', label='MI')
        plt.plot(np.arange(1, cir_num), X2_ave_accs, 'r', label='X2')
        plt.xlabel('Circulation Times')
        plt.ylabel('Prediction Accuracy')
        plt.xticks(np.arange(0, cir_num, (cir_num-1)/10))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(loc=0, ncol=1)
        plt.show()


        np.savetxt('cond_pre_accu',cond_ave_accs)
        np.savetxt('MI_pre_accu',MI_ave_accs)
        np.savetxt('X2_pre_accu',X2_ave_accs)
        np.savez_compressed('data/X2_pro.npz', arr = X2_pro)
        

    @ indexedproperty
    def trans_slot(self, key: float) -> float:
        return self._data.at[key, 'transmission slot']

    @ trans_slot.setter
    def trans_slot(self, key: int, value: float):
        self._data.at[key, 'transmission slot'] = value

    @ indexedproperty
    def location(self, key: int) -> int:
        return self._data.at[key, 'location']

    @ location.setter
    def location(self, key: int, value: int):
        if type(value) != int:
            raise TypeError('The input value type must be int.')

        self._data.at[key, 'location'] = str(value)

    @ indexedproperty
    def trans_prob(self, key: int) -> float:
        return self._data.at[key, 'transmission probability']

    @ trans_prob.setter
    def trans_prob(self, key: int, value: float):
        self._data.at[key, 'transmission probability'] = value


if __name__ == '__main__':
    sensor_contrib = SensorContrib()

    # 查看list
    # act_times, joint_times = sensor_contrib.circul(10)
    # # pri_pro, cond_pro, pri_pro_re, joint_pro  = sensor_contrib.cal_cond_pro(act_times, joint_times, 10)
    # # MI_pro = sensor_contrib.cal_MI(act_times, joint_times, 10)
    # X2_pro = sensor_contrib.cal_X2(act_times, joint_times, 10)
    # pre_accu, sel_sensor, sort_sensor, activated_t1, activated_t2 = sensor_contrib.cal_pre_accu(1,X2_pro)
    # # np.savetxt("cond_pro.csv", cond_pro, delimiter=",")
    # np.savetxt("sort_sensor.csv", sort_sensor, delimiter=",")
    # np.savetxt("sel_sensor.csv", sel_sensor, delimiter=",")
    # # np.savetxt("pri_pro.csv", pri_pro, delimiter=",")
    # # np.savetxt("MI_pro.csv", MI_pro, delimiter=",")
    # np.savetxt("X2_pro.csv", X2_pro, delimiter=",")
    # # np.savetxt("pxpy.csv", pxpy, delimiter=",")
    # np.set_printoptions(threshold= np.inf)
    # # print(pri_pro)
    # # print(cond_pro)
    # # print(sort_sensor)
    # print(activated_t1)
    # print(activated_t2)
    # # print(pre_accu)

    # # pre_accu = sensor_contrib.cal_pre_accu(1,cond_pro)
    # # print(pre_accu)
    # # plt.imshow(cond_pro)

    # 绘制三种概率的预测精度图
    cond_ave_accs = []
    MI_ave_accs = []
    X2_ave_accs = []
    act_times = np.zeros((sensor_contrib._length), 'int64')
    joint_times = np.zeros(
        (sensor_contrib._length, sensor_contrib._length), 'int64')
    sensor_contrib.plot_pic(act_times, joint_times, 11)
    # for i in range(1,501):
    #     act_times_temp, joint_times_temp = sensor_contrib.circul(1)
    #     act_times += act_times_temp
    #     joint_times += joint_times_temp
    #     pri_pro, cond_pro, pri_pro_re, joint_pro  = sensor_contrib.cal_cond_pro(act_times, joint_times, i)
    #     MI_pro = sensor_contrib.cal_MI(act_times, joint_times, i)
    #     X2_pro = sensor_contrib.cal_X2(act_times, joint_times, i)
    #     cond_ave_accs.append(sensor_contrib.cal_ave_pre_accu(0,9,cond_pro))
    #     MI_ave_accs.append(sensor_contrib.cal_ave_pre_accu(0,9,MI_pro))
    #     X2_ave_accs.append(sensor_contrib.cal_ave_pre_accu(0,9,X2_pro))
    # plt.plot(np.arange(1,501), cond_ave_accs, 'b')
    # plt.plot(np.arange(1,501), MI_ave_accs, 'g')
    # plt.plot(np.arange(1,501), X2_ave_accs, 'r')
    # plt.xlabel('Circulation Times')
    # plt.ylabel('Prediction Accuracy')
    # plt.xticks(np.arange(0,501,50))
    # plt.yticks(np.arange(0,1.1,0.1))
    # plt.show()
    # print(ave_accs)

    print('=> Generate done.')
