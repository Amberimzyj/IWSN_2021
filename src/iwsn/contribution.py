# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-21 15:27:10
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 13:16:23
# @License: MIT LICENSE

import os
import random
import math
from ast import literal_eval
from typing import List, Tuple, Union

from indexedproperty import indexedproperty
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from iwsn.utils.patterns import singleton
from iwsn.utils.naive_bayes import NavieBayes


@singleton
class SensorContrib(object):
    '''The sensor contribution class.'''

    def __init__(self,
                 data_path: str = 'data/sensors.csv',
                 active_thresh: float = 0.2,
                 sensor_num_pre_t: int = 10,
                 trans_time_interal: int = 3,
                 feature_sensor_dis: float = 3.,
                 bayes_type: str = 'MultinomialNB'):
        """The sensor conttibution class.

        Keyword Arguments:
            data_path {str} -- Path to load dataframe. (default: {'data/sensors.csv'})

        Raises:
            FileNotFoundError: The input data path not found.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f'data path "{data_path}" not exists.')

        self._data = pd.read_csv(data_path)
        self._active_th = active_thresh
        self._sensor_npt = sensor_num_pre_t
        self._n_timeslots = int(len(self._data) / self._sensor_npt)
        self._tti = trans_time_interal
        self._feat_sensor_dis = feature_sensor_dis

        self._path = data_path
        self._length = len(self._data)
        self._naive_bayes = NavieBayes(bayes_type)

    def static_stage(self, max_sensors: int = 10):
        if self._n_timeslots <= 2:
            raise ValueError("数据的timeslots必须大于2")

        # 获得前两个time slot激活的节点
        activated_t0 = self.t_activate(0)
        activated_t1 = self.t_activate(1)

        # ave_pp_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
        # ave_mi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
        # ave_chi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')

        # 依次遍历后续的time slot
        for t in range(2, self._n_timeslots - 1):
            # self._naive_bayes = NavieBayes("MultinomialNB")
            train_X_index, train_X, train_y = self.nb_gen_train_data(
                activated_t0, activated_t1)
            self._naive_bayes.fit(train_X_index, train_X,
                                  train_y, partial=True)

            eval_X_index, eval_X, eval_y, eval_length = self.select_sensor(
                activated_t1, t)
            predict_proba = self._naive_bayes.predict_proba(eval_X)[:, 0]
            predict_sensors = np.take(
                eval_X_index, np.argsort(predict_proba)[:max_sensors])

            activated_t0 = activated_t1
            activated_t1 = self.t_activate(t)

            acc = len(np.intersect1d(activated_t1, predict_sensors)
                      ) / len(activated_t1)
            print(f'slot: {t}, acc: {acc}')

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

    def _nb_cal_succ_predict_radio(self, t: int, max_sensors: int, X: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
        indexed_metrics = self._naive_bayes.select(X, y, metric)
        activated = np.array(self.t_activate(t))
        accs = self._cal_succ_predict_radio(indexed_metrics[:, 0], activated)
        filled_accs = np.full(max_sensors, accs[-1], 'float')
        filled_accs[:len(accs)] = accs[:max_sensors]

        return filled_accs

    def _cal_succ_predict_radio(self, predict, actual):
        predict = predict.astype('int')

        accs = np.empty(len(predict), 'float')
        for i in range(1, len(predict)+1):
            inter = np.intersect1d(predict[:i], actual)
            acc = len(inter) / len(actual)
            accs[i-1] = acc

        return accs

    def t_activate(self, t: int) -> List[int]:
        """Calculate the activate sensors at timeslot t.

        Arguments:
            t {int} -- the specified timeslot.

        Returns:
            List[int] -- Return the activated sensors' index.
        """
        activated = []
        for i in range(self._sensor_npt * t, self._sensor_npt * (t + 1)):
            if self.trans_prob[i] >= self._active_th:
                activated.append(i)

        return activated

    def require_distance(self, distance: int) -> bool:
        """判断两个节点间的距离是否小于阈值

        Args:
            distance (int): 输入的节点间距离

        Returns:
            bool: 返回判断结果
        """
        return distance < self._feat_sensor_dis and distance > 0

    def select_sensor(self, activated: list, t: int) -> Tuple[list, list, list, int]:
        """根据上一时隙激活的节点选择后续的节点

        Args:
            activated (list): 输入的上一时隙激活节点列表
            t (int): 当前时隙time slot

        Returns:
            Tuple[list, list, list, int]: 返回选择后的节点数据
        """
        length = 0

        data_X_index = []
        data_X = []
        data_y = []

        for y in activated:
            for x in range(self._sensor_npt * t, self._length):
                distance = self._cal_dis(self.location[x], self.location[y])
                if self.require_distance(distance):
                    length += 1
                    data_X_index.append(x)
                    data_X.append([distance])
                    data_y.append(y)

        return data_X_index, data_X, data_y, length

    def nb_gen_train_data(self, activated_t1: list, activated_t2: list) -> Tuple[list, list, list]:
        """生成朴素贝叶斯训练数据

        Args:
            activated_t1 (list): 前一个时刻触发的节点集
            activated_t2 (list): 当前时刻触发的节点集

        Returns:
            Tuple[list, list, list]: 训练的数据
        """
        data_X_index = []
        data_X = []
        data_y = []

        start_y = max(activated_t2)
        for y in activated_t1:
            for x in activated_t2:
                distance = self._cal_dis(self.location[x], self.location[y])
                data_X_index.append(x)
                data_X.append([distance])
                data_y.append(1)

            # for x in random.sample(range(start_y+1, self._length), len(activated_t2)):
            for x in range(start_y+1, self._length):
                distance = self._cal_dis(self.location[x], self.location[y])
                if self.require_distance(distance):
                    data_X_index.append(x)
                    data_X.append([distance])
                    data_y.append(0)

        return data_X_index, data_X, data_y

    def save_data(self):
        '''Save dataframe to csv.'''
        self._data.to_csv(self._path)

    def _cal_dis(self, x: int, y: int) -> float:
        """计算两个节点间的距离

        Args:
            x (int): 节点x的位置
            y (int): 节点y的位置

        Raises:
            TypeError: 输入的位置类型错误
            ValueError: 输入的位置值错误

        Returns:
            float: 返回计算后的距离
        """
        for a in (x, y):
            if a < 0:
                raise ValueError("输入的位置必须大于等于0")

        return int((x - y) / 10)

    @indexedproperty
    def trans_slot(self, key: float) -> float:
        return self._data.at[key, 'transmission slot']

    @trans_slot.setter
    def trans_slot(self, key: int, value: float):
        self._data.at[key, 'transmission slot'] = value

    @indexedproperty
    def location(self, key: int) -> int:
        return self._data.at[key, 'location']

    @location.setter
    def location(self, key: int, value: int):
        if type(value) != int:
            raise TypeError('The input value type must be int.')

        self._data.at[key, 'location'] = str(value)

    @indexedproperty
    def trans_prob(self, key: int) -> float:
        return self._data.at[key, 'transmission probability']

    @trans_prob.setter
    def trans_prob(self, key: int, value: float):
        self._data.at[key, 'transmission probability'] = value


if __name__ == '__main__':
    sensor_contirb = SensorContrib()
    sensor_contirb.static_stage()
