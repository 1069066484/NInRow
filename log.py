# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: definition of logger.
"""
import logging
import data_utils 
from global_defs import *
import sys



def init_logger(name, fn):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(join(FOLDER_LOGS, fn + '_'+data_utils.curr_time_str()+'.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    return logger



class Logger(object):
    def __init__(self, file=None, console=True, flush_freq=10):
        self.terminal = sys.stdout
        self.file = None if file is None else open(file, "w")
        self.console = console
        self.flush_freq = flush_freq
        self.write_cnt = 0

    def log(self, *message):
        message = (' '.join([str(m) for m in message])) + '\n'
        if self.console:
            self.terminal.write(message)
        if self.file is not None:
            self.file.write(message)
            self.write_cnt += 1
            if self.write_cnt % self.flush_freq == 0:
                self.file.flush() 
 
    def flush(self):
        if self.file is not None:   
            self.file.flush() 