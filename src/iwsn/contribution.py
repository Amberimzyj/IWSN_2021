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
from typing import List, Union

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
                 active_thresh: float = 0.7,
                 sensor_num_pre_t: int = 100,
                 trans_time_interal: int = 3,
                 feature_sensor_time: float = 0.5,
                 feature_sensor_dis: float = 5.,
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
        self._feat_sensor_time = feature_sensor_time
        self._feat_sensor_dis = feature_sensor_dis

        self._path = data_path
        self._length = len(self._data)
        self._naive_bayes = NavieBayes(bayes_type)

    def static_stage(self, max_sensors: int = 100):
        activated = self.t_activate(0)

        ave_pp_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
        ave_mi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')
        ave_chi_accs = np.zeros((self._n_timeslots-2, max_sensors), 'float')

        for t in tqdm(range(1, self._n_timeslots - 1)):
            selected, sel_length = self.select_sensor(activated, t)
            data_X_index, data_X, data_y = self.nb_make_dataset(
                selected, sel_length)

            if len(data_X) == 0:
                activated = np.array(self.t_activate(t))
                continue

            self._naive_bayes.fit(data_X_index, data_X, data_y)
            ave_pp_accs[t-1] = self._nb_cal_succ_predict_radio(
                t, max_sensors, data_X, data_y, 'posterior_prob')
            ave_mi_accs[t-1] = self._nb_cal_succ_predict_radio(
                t, max_sensors, data_X, data_y, 'mutual_information')
            ave_chi_accs[t-1] = self._nb_cal_succ_predict_radio(
                t, max_sensors, data_X, data_y, 'chi_square_test')

        ave_pp_accs = ave_pp_accs.mean(axis=0)
        ave_mi_accs = ave_mi_accs.mean(axis=0)
        ave_chi_accs = ave_chi_accs.mean(axis=0)

        plt.plot(ave_pp_accs, color='green')
        plt.plot(ave_mi_accs, color='blue')
        plt.plot(ave_chi_accs, color='red')
        plt.show()

    def _nb_cal_succ_predict_radio(self, t: int, max_sensors: int, X: np.ndarray, y: np.ndarray, metric: str):
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
        for i in range(self._sensor_npt * t, self._sensor_npt * (t + 1)):   # 10 minislots
            if self.trans_prob[i] >= self._active_th:
                self.trans_time[i] = random.uniform(
                    self._tti * t, self._tti * (t + 1))
                activated.append(i)

        return activated

    def select_sensor(self, activated: list, t: int):
        """Select the feature sensors base on the acticated sensors list.

        Arguments:
            activated {list} -- The activated sensots list.
            t {int} -- The calculate time slot.

        Returns:
            (dict, int) -- The selected sensors dict and sensors' length
        """

        def require_activated():
            return (self.trans_time[x] != 0
                    and self.sensor_type[x] == self.sensor_type[y]
                    and abs(self.trans_time[x] - self.trans_time[y]) < self._feat_sensor_time)

        def require_distance():
            return self._euclidean_dis(self.location[x], self.location[y]) < self._feat_sensor_dis

        selected = {}
        length = 0

        for y in activated:
            selected[y] = []
            for x in range(self._sensor_npt * t, self._sensor_npt * (t + 1)):
                if require_activated() or require_distance():
                    length += 1
                    selected[y].append(x)

        return selected, length

    def nb_make_dataset(self, selected: dict, length: int):
        """Convert dataset to sklearn style data.

        Arguments:
            selected {dict} -- The selected sensors dict.
            length {int} -- The number of selected sensors.

        Returns:
            (np.ndarray, np.ndarray) -- The converted data X and y.
        """
        data_X_index = np.empty(length, 'int')
        data_X = np.empty((length, 2), 'int')
        data_y = np.empty(length, 'int')

        count = 0
        for y, feat_sensors in selected.items():
            for x in feat_sensors:
                data_X_index[count] = x
                data_X[count] = self.sensor_type[x], round(
                    self._euclidean_dis(self.location[x], self.location[y]))
                data_y[count] = y
                count += 1

        return data_X_index, data_X, data_y

    def save_data(self):
        '''Save dataframe to csv.'''
        self._data.to_csv(self._path)

    def _euclidean_dis(self, x: Union[tuple, list], y: Union[tuple, list]):
        for a in (x, y):
            if not isinstance(a, (tuple, list)):
                raise TypeError("The points' type must be tuple or list.")
            if len(a) == 0:
                raise ValueError("The points' must have at least one value.")

        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

    @indexedproperty
    def sensor_type(self, key: int) -> int:
        return self._data.at[key, 'sensor type']

    @sensor_type.setter
    def sensor_type(self, key: int, value: int):
        self._data.at[key, 'sensor type'] = value

    @indexedproperty
    def trans_time(self, key: float) -> float:
        return self._data.at[key, 'transmission time']

    @trans_time.setter
    def trans_time(self, key: int, value: float):
        self._data.at[key, 'transmission time'] = value

    @indexedproperty
    def location(self, key: int) -> tuple:
        return literal_eval(self._data.at[key, 'location'])

    @location.setter
    def location(self, key: int, value: tuple):
        if type(value) != tuple:
            raise TypeError('The input value type must be tuple.')

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
