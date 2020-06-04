# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-21 15:27:10
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 13:16:23
# @License: MIT LICENSE

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
            n=5000,
            ranges=[1, 6],
            sensor_list=[1, 2, 3, 4, 5],
            save_path="data/sensors.csv"):
        self._length = n

        process = self._gen_process(ranges)
        sensor_type = self._gen_sensor_type(sensor_list)
        trans_time = self._gen_trans_time()
        location = self._gen_loaction(process)
        trans_probs = self._gen_trans_prob()

        dataframe = pd.DataFrame({"technological process": process,
                                  "sensor type": sensor_type,
                                  "transmission time": trans_time,
                                  "location": location,
                                  "transmission probability": trans_probs})
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

    def _gen_trans_time(self):
        return np.zeros(self._length, 'float')

    def _gen_loaction(self, process: List[int]) -> List[tuple]:
        locations = []
        for i, pcs in enumerate(process):
            x = random.randint(20 * (pcs - 1), 20 * pcs)
            y = random.randint(0, 20)
            locations.append((x, y))

        return locations

    def _gen_trans_prob(self) -> List[float]:
        trans_probs = [random.uniform(0, 1) for i in range(self._length)]

        return trans_probs


if __name__ == '__main__':
    generator = SensorGen()
    generator.gen()
    print('=> Generate done.')
