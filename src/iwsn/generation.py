# Project: iwsn
# File Created: Friday, 26th June 2020 7:21:59 pm
# Author: Yajing Zhang (amberimzyj@qq.com)
# -----
# Last Modified: Friday, 26th June 2020 7:34:49 pm
# Modified By: Yajing Zhang (amberimzyj@qq.com>)
# -----
# Copyright 2017 - 2020 Your Company, Your Company





import math
import random
from typing import Union, List

import numpy as np
import pandas as pd

from iwsn.utils.patterns import singleton


@singleton
class SensorGen(object):
    ''' The sensor generation class.'''

    def gen(self,
            n=100,
            ranges=[1, 6],
            sensor_list=[1, 2, 3, 4, 5],
            save_path="data/sensors.csv",
            sensors_everyslot=10,  # 每个timeslot的sensor数量=钢板速度
            act_ratio=0.8
            ):  # 每个timeslot激活传感器数量的比例
        self._length = n
        self._sensornum = sensors_everyslot  # 每个timeslot的sensor数量
        self._act_num = int(self._sensornum * act_ratio)  # 每个timeslot激活的传感器数量

        process = self._gen_process(ranges)
        time_slot = self._gen_time_slot()
        sensor_type = self._gen_sensor_type(sensor_list)
        trans_time = self._gen_trans_time()
        location = self._gen_loaction(time_slot)
        trans_probs = self._gen_trans_prob()
        # select_sensor = self._gen_select_sensor()

        dataframe = pd.DataFrame({  # "technological process": process,
            "time slot": time_slot,
            # "sensor type": sensor_type,
            # "transmission time": trans_time,
            "location": location,
            "transmission probability": trans_probs,
            # "selected sensor": select_sensor
        })
        dataframe.to_csv(save_path)

    def _gen_process(self, ranges: Union[tuple, list]) -> List[int]:
        if not isinstance(ranges, (tuple, list)):
            raise Exception(
                f'ranges:{ranges} must be a instance of tuple or list.')
        if len(ranges) != 2:
            raise Exception('The length of ranges not equal to 2.')
        if ranges[1] <= ranges[0]:
            raise Exception('ranges[1] must lager than ranges[0].')

        iters = math.ceil(self._length / (ranges[1] - ranges[0]))
        process = [a for a in range(*ranges) for i in range(iters)]

        return process

    def _gen_sensor_type(self, sensor_list: list) -> list:
        if not isinstance(sensor_list, list):
            raise Exception('The sensor_list must be a list.')
        if len(sensor_list) == 0:
            raise Exception('The sensor_list length must larger than 0.')

        sensor_type = random.choices(sensor_list, k=self._length)

        return sensor_type

    def _gen_time_slot(self) -> list:
        serial = math.ceil(self._length/self._sensornum)  # 得到timeslot个数
        timeslot = [a for a in range(serial)
                    for i in range(self._sensornum)]
        return timeslot

    def _gen_trans_time(self):
        return np.zeros(self._length, 'float')

    def _gen_loaction(self, time_slot: List[int]) -> List[int]:
        locations = []
        for pcs in time_slot:
            x = random.randint(10*pcs, 10*(pcs+1))
            #x = random.randint( 10*(pcs - 1), 10*pcs)
            #y = random.randint(0, 10)
            #locations.append((x, y))
            locations.append(x)

        return locations

    def _gen_trans_prob(self) -> List[float]:
        trans_probs = [random.uniform(0, 1) for i in range(self._length)]

        return trans_probs

    def _gen_select_sensor(self) -> list:
        select_sensor = [0]*self._length
        serial = math.ceil(self._length/self._sensornum)  # 得到timeslot个数
        for i in range(serial):
            for j in range(10*i, 10*i+self._act_num):
                select_sensor[j] = 1
        return select_sensor


if __name__ == '__main__':
    generator = SensorGen()
    generator.gen()
    print('=> Generate done.')
