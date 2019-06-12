# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: some global definitions. This script should
            include no other scripts in the project.
"""

import os
from enum import IntEnum


join = os.path.join
exists = os.path.exists
def mkdir(dir):
    if not exists(dir):
        os.mkdir(dir)
    return dir


FOLDER_LOGS = mkdir('logs')
FOLDER_ZERO_NNS = mkdir('zero_nns')


if __name__=='__main__':
    pass

