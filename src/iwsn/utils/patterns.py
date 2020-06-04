# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-21 15:27:10
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 13:16:23
# @License: MIT LICENSE


def singleton(cls):
    # Dict to store instance object.
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)

        return _instance[cls]

    return _singleton
