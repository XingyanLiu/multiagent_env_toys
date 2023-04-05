# -*- coding: UTF-8 -*-
"""
@Author: Xingyan Liu
@CreateDate: 2023-04-05
@File: __init__.py
@Project: RLToys
"""
import os
import sys
from pathlib import Path
from typing import Union, Optional, Sequence, Mapping
import time
import logging
import numpy as np
import pandas as pd


# %matplotlib inline
# # 以下两行是为了在模块被修改之后重新载入
# %load_ext autoreload 
# %autoreload 2


def __test__():
    pass


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    t = time.time()

    __test__()

    print('Done running file: {}\nTime: {}'.format(
        os.path.abspath(__file__), time.time() - t,
    ))
