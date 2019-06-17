"""
@Author: Zhixin Ling
@Description: A general and flexible CNN model. 
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import debug as tf_debug
import numpy as np
from sklearn.metrics import *
from data_utils import *
from global_defs import *
from tensorflow.contrib.slim import nets



class CNN:
    class Params:
        """
        cnns:   a list with scalars(pooling layer) or two-element list(channels, kernel_size) as its element
        fcs:    a list of scalars, indicating neurons of fully connected layers. Set fcs to None if you don't want the
              output flattened.
        """
        def __init__(self, cnns, fcs):
            self.cnns = cnns
            self.fcs = fcs

        def __str__(self):
            return '[cnn:' + str(self.cnns) + '   fcs:' + str(self.fcs) + ']'

        def construct(self, input, keep_prob=1.0, scope_prefix=""):
            conv_cnt = 1
            pool_cnt = 1
            net = input
            for param in self.cnns:
                if isinstance(param, int):
                    net = slim.max_pool2d(net, [param,param], scope=scope_prefix + 'pool' + str(pool_cnt))
                    pool_cnt += 1
                else:
                    net = slim.conv2d(net, param[0], [param[1],param[1]], scope=scope_prefix+'conv' + str(conv_cnt))
                    conv_cnt += 1
            if self.fcs is None:
                return net
            net = slim.flatten(net)
            for idx, param in enumerate(self.fcs):
                net = slim.fully_connected(net, param, scope=scope_prefix+'fc' + str(idx))
                net = slim.dropout(net, keep_prob=keep_prob)
            return net

    def __init__(self, cnn_params=Params([[16,5],2,[32,5],2], [1024]), 
                 kp=0.5, lr_init=0.05, lr_dec_rate=0.95, batch_size=128,
                 epoch=10, verbose=False, act=tf.nn.relu, l2=5e-8, path=None,
                 resnet_v2=None):
        """
        built_net: a resnet_v2. Like nets.resnet_v2.resnet_v2_50. Check source codes of tensorflow.contrib.slim.nets.resnet_v2
            for how to make a resnet! The input images first go through built_net and then the net constructed by CNN. 
        """
        self.params = cnn_params
        self.kp = kp
        self.lr_init = lr_init
        self.lr_dec_rate = lr_dec_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.act = act
        self.l2 = l2
        self.path = None if path is None else mkdir(path)
        self.sess = None
        self.resnet_v2 = resnet_v2
        self.ts = {}
        self.var_names = ['kp', 'y', 'acc', 'is_train', 'pred', 'global_step', 'loss','x', 'train_step']

    def print_vars(self):
        variable_names = tf.global_variables()
        for name in variable_names:
            print(name)
        op = self.graph.get_operations()
        for i in op:
            print(i)

    def init_vars(self):
        for ts in self.var_names:
                self.ts[ts] = tf.get_collection(ts)[0]

    def __str__(self):
        return "CNN-- structure: {} \tkp: {} \tlr_init: {} \tlr_dec_rate: {} \tbatch_size: {} \tepoch: {} \tact: {}".format(
            self.params, self.kp, self.lr_init, self.lr_dec_rate, self.batch_size, self.epoch, str(self.act).split(' ')[1] if self.act is not None else 'NONE')

    def init_training_data(self, X, Y, reserve_test):
        self.Y_min = np.min(Y)
        if reserve_test is not None:
            xy_tr, xy_te = labeled_data_split([X, Y], 1.0-reserve_test)
            X, Y = xy_tr
            X_te, Y_te = xy_te
            self.X_te = X_te
            self.Y_te = labels2one_hot(Y_te) + self.Y_min
        else:
            self.X_te = None
            self.Y_te = None
        self.X = X
        self.Y = labels2one_hot(Y) + self.Y_min

    def fit(self, X, Y, reserve_test=None, refresh_saving=False):
        """
        If you wanna extract test set automatically, set reserve_test the ratio for test set
        """
        self.init_training_data(X, Y, reserve_test)
        self.construct_model()
        self.init_sess(refresh_saving)
        self.train()

    def construct_model(self):
        tf.reset_default_graph()
        n_xs, slen = self.X.shape
        slen = int(round(np.sqrt(slen)))
        n_labels = self.Y.shape[1]
        x = tf.placeholder(tf.float32, [None, slen*slen], name='x')
        x_trans = tf.reshape(x, [-1, slen, slen, 1])
        if self.resnet_v2 is not None:
            x_trans = tf.image.grayscale_to_rgb(x_trans)
        kp = tf.placeholder(tf.float32, [], name='kp')
        y = tf.placeholder(tf.float32, [None, n_labels], name='y')
        is_train = tf.placeholder(tf.bool, [], name='is_train')
        net = x_trans
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=self.act,
                    normalizer_fn=tf.layers.batch_normalization,
                    normalizer_params={'training': is_train, 'momentum': 0.95},
                    weights_regularizer=slim.l2_regularizer(self.l2)):
            if self.resnet_v2 is not None:
                net, _ = self.resnet_v2(
                net, num_classes=None, is_training=is_train, global_pool=True)
            if self.params is not None:
                net = self.params.construct(net, kp)
            if len(net.shape) > 2:
                net = slim.flatten(net)
            logits = slim.fully_connected(net, n_labels, activation_fn=None, scope='logits')
        pred = tf.argmax(logits,1, name='pred')
        corrects = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
        acc = tf.reduce_mean(tf.cast(corrects, tf.float32),name='acc')
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        loss = tf.add(cross_entropy, regularization_loss, name='loss')
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        lr = tf.train.exponential_decay(
            self.lr_init,
            global_step,
            n_xs / self.batch_size, self.lr_dec_rate,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = slim.learning.create_train_op(
            loss,  optimizer, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_step = control_flow_ops.with_dependencies([updates], train_step)
        locs = locals()
        for var in self.var_names:
            tf.add_to_collection(var, locs[var])

    def next_batch(self):
        batch_sz = self.batch_size
        indices = list(range(self.curr_tr_batch_idx, self.curr_tr_batch_idx+batch_sz))
        self.curr_tr_batch_idx = (batch_sz + self.curr_tr_batch_idx) % self.X.shape[0]
        indices = [i%self.X.shape[0] for i in indices]
        return [self.X[indices], self.Y[indices]]

    def run_acc(self, X, Y):
        correct_preds = 0.0
        for batch_idx in range(0,X.shape[0],self.batch_size):
            batch_idx_next = min(X.shape[0], batch_idx + self.batch_size)
            batch_xs = X[batch_idx:batch_idx_next]
            batch_ys = Y[batch_idx:batch_idx_next]
            acc = self.sess.run(self.ts['acc'],feed_dict=
                    {self.ts['x']: batch_xs, self.ts['kp']: 1.0, self.ts['is_train']: False, self.ts['y']: batch_ys})
            #print(acc, acc * (batch_idx_next - batch_idx), X.shape)
            correct_preds += acc * (batch_idx_next - batch_idx)
        return correct_preds / X.shape[0]

    def init_sess(self, refresh_saving):
        """
        return whether use new parameters
        """
        if exists(join(self.path, '0.meta')):
            tf.reset_default_graph()
            sess = tf.Session()
            self.saver = tf.train.import_meta_graph(join(self.path, '0.meta'))
            print("Find the meta in file", self.path)
        else:
            print("Init new meta")
            self.saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        self.init_vars()
        self.sess = sess
        if not refresh_saving and self.path is not None: 
            try: 
                self.saver.restore(sess,tf.train.latest_checkpoint(self.path)) 
                print("Find the lastest check point in file", self.path)
                return True
            except: 
                print("Init new parameters")
                return False

    def train(self):
        sess = self.sess
        self.saver.save(sess, join(self.path, '0'), write_meta_graph=True)
        self.curr_tr_batch_idx = 0
        it_pep = round(self.X.shape[0] / self.batch_size)
        x_t = self.ts['x']; kp_t = self.ts['kp']; y_t = self.ts['y']; is_train_t = self.ts['is_train']; 
        train_step_t = self.ts['train_step']; global_step_t = self.ts['global_step']
        for i in range(round(self.epoch * self.X.shape[0] / self.batch_size)+1):
            batch_xs, batch_ys = self.next_batch()
            feed_dict = {x_t: batch_xs, kp_t: self.kp, y_t: batch_ys, is_train_t: True}
            sess.run(train_step_t, feed_dict=feed_dict)
            global_step = sess.run(global_step_t, feed_dict=feed_dict)
            if self.verbose and global_step % it_pep == 0:
                print("iteration:",i,' global_step:',global_step, '  train_acc: ',self.run_acc(self.X, self.Y), '   test_acc:', 
                      -1.0 if self.X_te is None else self.run_acc(self.X_te, self.Y_te))
                if self.path is not None:
                    self.saver.save(sess, self.path + '/model', global_step=global_step_t, write_meta_graph=False)

    def predict(self, X):
        if self.sess is None:
            if not self.init_sess(False):
                raise Exception("Error: trying to predict without trained network")
        pred = self.sess.run(self.ts['pred'], feed_dict={self.ts['x']: X, self.ts['kp']: 1.0, self.ts['is_train']: False})
        return pred


def main_mnist():
    data, labels = read_mnist_dl()
    # print(data.shape)
    data = data[:5000]
    labels = labels[:5000]
    cnn = CNN(path='log_noresCNN',epoch=3, verbose=True, batch_size=4, resnet_v2=None)
    cnn.fit(data, labels, 0.2)


def main_mnist_res():
    data, labels = read_mnist_dl()
    # print(data.shape)
    data = data[:5000]
    labels = labels[:5000]
    cnn = CNN(path='log_resCNN',epoch=3, verbose=True, batch_size=4, resnet_v2=nets.resnet_v2.resnet_v2_50, cnn_params=None)
    cnn.fit(data, labels, 0.2)


if __name__ == "__main__":
    # main_mnist()
    main_mnist_res()
