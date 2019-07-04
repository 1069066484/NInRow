
# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of NinRowAI: some help functions.
"""

from global_defs import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import csv
from sklearn.manifold import TSNE
import gzip
from scipy.io import loadmat
import pickle
import datetime


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def npfilename(filename):
    return posfix_filename(filename, '.npy')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def csvfilename(filename):
    return posfix_filename(filename, '.csv')


def h5filename(filename):
    return posfix_filename(filename, '.h5')


def logfilename(filename):
    return posfix_filename(filename, '.log')


def txtfilename(filename):
    return posfix_filename(filename, '.txt')


npfn = npfilename
pkfn = pkfilename
csvfn = csvfilename
h5fn = h5filename
logfn = logfilename
txtfn = txtfilename


def curr_time_str():
    return datetime.datetime.now().strftime('%y%m%d%H%M%S')


def read_mnist(one_hot=True):
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(FOLDER_MNIST,one_hot=one_hot)


def read_mnist_dl():
    path_data = join(FOLDER_MNIST, npfn('data'))
    path_labels = join(FOLDER_MNIST, npfn('labels'));
    if exists(path_data):
        return [np.load(path_data), np.load(path_labels)]
    mnist = read_mnist()
    dls = [np.vstack([mnist.train.images, mnist.test.images]),
           np.vstack([mnist.train.labels, mnist.test.labels])]
    np.save(path_data, dls[0])
    np.save(path_labels, dls[1])
    return dls


def csvfile2nparr(csvfn, cols=None):
    csvfn = csvfilename(csvfn)
    csvfn = csv.reader(open(csvfn,'r'))
    def read_line(line):
        try:
            # return [float(i) for i in line if cols is None or i in cols]
            # print(cols)
            return [float(line[i]) for i in range(len(line)) if cols is None or i in cols]
        except:
            return None
    m = [read_line(line) for line in csvfn]
    m = [l for l in m if l is not None]
    return np.array(m)


def read_labeled_features(csvfn):
    arr = csvfile2nparr(csvfn)
    data, labels = np.hsplit(arr,[-1])
    labels = labels.reshape(labels.size)
    return [data, labels]



def plt_show_it_data(it_data, xlabel='iterations', ylabel=None, title=None, do_plt_last=True):
    y = it_data
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    if do_plt_last:
        plt.text(x[-1], y[-1], y[-1])
    plt.show()


def plt_show_scatter(xs, ys, xlabel=None, ylabel=None, title=None):
    colors = ['r', 'y', 'k', 'g', 'b', 'm']
    num2plt = min(len(colors), len(xs))
    for i in range(num2plt):
        plt.scatter(x=xs[i], y=ys[i], c=colors[i], marker='.')
    plt.xlabel('' if xlabel is None else xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    plt.show()


def non_repeated_random_nums(nums, num):
    num = math.ceil(num)
    nums = np.random.permutation(nums)
    return nums[:num]


def index_split(num, percent1):
    percent1 = math.ceil(num * percent1)
    nums = np.random.permutation(num)
    return [nums[:percent1], nums[percent1:]]


def labeled_data_split(labeled_data, percent_train=0.6):
    np.random.seed(0)
    train_idx, test_idx = index_split(labeled_data[0].shape[0], percent_train)
    #train_ld = [labeled_data[0][train_idx], labeled_data[1][train_idx]]
    #test_ld = [labeled_data[0][test_idx], labeled_data[1][test_idx]]
    train_ld = [ld[train_idx] for ld in labeled_data]
    test_ld = [ld[test_idx] for ld in labeled_data]
    return [train_ld, test_ld]


def rand_arr_selection(arr, num):
    nonrep_rand_nums = non_repeated_random_nums(arr.shape[0], num)
    return [arr[nonrep_rand_nums], nonrep_rand_nums]


def labels2one_hot(labels):
    labels = np.array(labels, dtype=np.int)
    if len(labels.shape) == 1:
        minl = np.min(labels)
        labels -= minl
        maxl = np.max(labels) + 1
        r = range(maxl)
        return np.array([[1 if i==j else 0 for i in r] for j in labels])
    return labels


def shuffle_labeled_data(dl):
    data, labels = dl
    a = np.arange(labels.shape[0])
    np.random.seed(0)
    np.random.shuffle(a)
    return [data[a], labels[a]]



def _test_labels_one_hot():
    a = np.array([2,1,0,0,0,2,1,1,1])
    print(labels2one_hot(a))


def raw_pendigits2dl(fn):
    labels = []
    data = []
    init_arr = np.zeros(16)
    for l in open(fn):
        l = l.split()
        data.append(init_arr.copy())
        labels.append(int(l[0]))
        for str in l[1:]:
            pos, val = str.split(':')
            data[-1][int(pos)-1] = float(val)/100.0
    return [np.array(data, dtype=np.float32), np.array(labels, dtype=np.int8)]


def test_non_repeated_random_nums():
    nums = non_repeated_random_nums(10,round(10 * 0.3))
    print(nums)
    a = np.array([9,8,7,6,5,4,3,2,1,0])
    l = [9,8,7,6,5,4,3,2,1,0]
    print(a[nums])
    print(l[nums])


def test_read_mnist_dl():
    dls = read_mnist_dl()
    print(dls[0].shape)
    print(dls[1].shape)


if __name__ == '__main__':
    test_read_mnist_dl()
    pass
    # print(__file__)
    # print(exists('./datasets_raw'))