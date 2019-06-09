# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some global definitions. This script should
            include no other scripts in the project.
"""

import os
from enum import IntEnum


join = os.path.join
exists = os.path.exists


FOLDER_LOGS = mkdir('logs')


def mkdir(dir):
    if not exists(dir):
        os.mkdir(dir)
    return dir


if __name__=='__main__':
    pass

