# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of NinRowAI: definition of dataset
"""
import numpy as np


class Dataset:
    def __init__(self, data, batch_size):
        self.curr_idx = 0
        self.epoch = 0
        self.data_len = len(data[0])
        self.batch_size = batch_size
        self.data = data

    def next_bach(self):
        while True:
            for batch in self.batches():
                return batch


    def batches(self):
        data = self.data
        for batch_idx in range(0,self.data_len,self.batch_size):
            batch_idx_next = batch_idx + self.batch_size
            # acuiqre the test batches
            if batch_idx_next >= self.data_len:
                self.epoch += 1
                yield [np.vstack([data[i][batch_idx:self.data_len], data[i][:batch_idx_next % self.data_len]])\
                   for i in range(len(data))]
                batch_idx_next -= self.data_len
            else:
                yield [data[i][batch_idx:batch_idx_next] for i in range(len(data))]
        self.epoch += 1



if __name__=='__main__':
    a = np.zeros([109, 13])
    print(a.shape)
    ds = Dataset([a], 64)
    for b in ds.batches():
        print(b[0].shape)
