# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: some global definitions. This script should include no other script in the project.
"""

import os
from enum import IntEnum
from utils.data import asdir, b2s, join, exists, mkdir


FOLDER_LOGS = mkdir('logs')
FOLDER_ZERO_NNS = mkdir('zero_nns')

FOLDER_SAVING = mkdir('saving')
FOLDER_MNIST = os.path.join(FOLDER_SAVING, 'MNIST_data')


if __name__=='__main__':
    pass

