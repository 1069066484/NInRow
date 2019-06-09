# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: network model. refer to AlphaGo Zero's network. One way in and two ways out.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import control_flow_ops


class ZeroNN:
    def __init__(self, cnns=[[16,5],2,[32,5],2], 
                 fcs=[1024], kp=0.5, lr_init=0.05, 
                 lr_dec_rate=0.95, batch_size=128,
                 epoch=10, verbose=False, act=tf.nn.relu,
                 l2=5e-8):
        self.cnns = cnns
        self.fcs = fcs
        self.kp = kp
        self.lr_init = lr_init
        self.lr_dec_rate = lr_dec_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.act = act
        self.l2 = l2

    def __str__(self):
        return "ZNN-- cnns: {} \tfcs: {} \tkp: {} \tlr_init: {} \tlr_dec_rate: {} \tbatch_size: {} \tepoch: {} \tact: {}".format(
            self.cnns, self.fcs, self.kp, self.lr_init, self.lr_dec_rate, self.batch_size, self.epoch, str(self.act).split(' ')[1] if self.act is not None else 'NONE'
            )

    def fit(self, X, Y):
        self.Y_min = np.min(Y)
        self.X = X
        self.Y = labels2one_hot(Y)
        self.construct_model()
        self.train()

    def construct_model(self):
        tf.reset_default_graph()
        n_xs, slen = self.X.shape
        slen = int(round(np.sqrt(slen)))
        n_labels = self.Y.shape[1]
        x_raw = tf.placeholder(tf.float32, [None, slen*slen])
        kp = tf.placeholder(tf.float32, [])
        y = tf.placeholder(tf.float32, [None, n_labels])
        x = tf.reshape(x_raw, [-1, slen, slen, 1])
        is_train = tf.placeholder(tf.bool, [])
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=self.act,
                    normalizer_fn=tf.layers.batch_normalization,
                    normalizer_params={'training': is_train, 'momentum': 0.95},
                    weights_regularizer=slim.l2_regularizer(self.l2)):
            conv_cnt = 1
            pool_cnt = 1
            net = x
            for param in self.cnns:
                if isinstance(param, int):
                    net = slim.max_pool2d(net, [param,param], scope='pool' + str(pool_cnt))
                    pool_cnt += 1
                else:
                    net = slim.conv2d(x, param[0], [param[1],param[1]], scope='conv' + str(conv_cnt))
                    conv_cnt += 1
            net = slim.flatten(net)
            for idx, param in enumerate(self.fcs):
                net = slim.fully_connected(net, param, scope='fc' + str(idx))
                net = slim.dropout(net, keep_prob=kp)
            logits = slim.fully_connected(net, n_labels, activation_fn=None, scope='logits')
        corrects = tf.equal(tf.arg_max(logits,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(corrects, tf.float32))
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        lr = tf.train.exponential_decay(
                self.lr_init,
                global_step,
                n_xs / self.batch_size, self.lr_dec_rate,
                staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = slim.learning.create_train_op(cross_entropy + regularization_loss, optimizer, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_step = control_flow_ops.with_dependencies([updates], train_step)
        self.x_t = x_raw
        self.kp_t = kp
        self.y_t = y
        self.acc_t = acc
        self.train_step_t = train_step
        self.is_train_t = is_train
        self.pred_t = tf.arg_max(logits,1)
        self.global_step = global_step

    def next_batch(self):
        batch_sz = self.batch_size
        indices = list(range(self.curr_tr_batch_idx, self.curr_tr_batch_idx+batch_sz))
        self.curr_tr_batch_idx = (batch_sz + self.curr_tr_batch_idx) % self.X.shape[0]
        indices = [i%self.X.shape[0] for i in indices]
        return [self.X[indices], self.Y[indices]]

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.curr_tr_batch_idx = 0
        it_pep = round(self.X.shape[0] / self.batch_size)
        for i in range(round(self.epoch * self.X.shape[0] / self.batch_size)+1):
            batch_xs, batch_ys = self.next_batch()
            feed_dict = {self.x_t: batch_xs,
                         self.kp_t: self.kp,
                         self.y_t: batch_ys,
                         self.is_train_t: True}
            sess.run(self.train_step_t, feed_dict=feed_dict)
            if self.verbose and sess.run(self.global_step, feed_dict=feed_dict) % it_pep == 0:
                print("iteration",i, '  train_acc: ',sess.run(self.acc_t,feed_dict={
                    self.x_t: batch_xs, self.kp_t: 1.0, self.is_train_t: False, self.y_t: batch_ys
                    }))
        self.sess = sess

    def predict(self, X):
        pred = self.sess.run(self.pred_t, feed_dict={self.x_t: X, self.kp_t: 1.0, self.is_train_t: False})
        return pred + self.Y_min



if __name__=='__main__':
    pass
