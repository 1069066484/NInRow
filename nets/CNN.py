"""
@Author: Zhixin Ling
@Description: A general and flexible CNN model. fully connected layers, common convolution layers and residual and inception blocks are supported.
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.python import debug as tf_debug
import numpy as np
from sklearn.metrics import *
from utils.data import *
from global_defs import *
from tensorflow.contrib.slim import nets
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import math_ops


class CNN:
    class Params:
        """
        @cnns:   a list with scalars(pooling layer) or two-element list(channels, kernel_size) as its element, 
                or use a list in list to indicate a residual block, set a kernel size negative to set paddings valid. 
                Set pooling layer scalar negative for average pooling. ResnetV2 is implemented.
            ex. [[16,5],2,[32,5],2] or [[16,5],2,[[32,5],[32,-5]], 2]
                set a layer's first element to None to indicate this is a parallel module. The expression below means a inception module.
            ex. 
            [
            [128,3],2,
            [
            [None,
                    [[64,1]], 
                    [[96,1],[128,3]],
                    [[96,1],[128,3], [192,5]],
                    ],
            ]], some kind of inception.
            [
            [16,5],2,
            [
            [None,[[32,5],[32,-5]], [[32,3],[32,3]]],
            [64,3],[64,3]
            ]], hard to tell what the structure is
            ATTENTION: the codes do not help verify the size of the features.
        @fcs:    a list of scalars, indicating neurons of fully connected layers. Set fcs to None if you don't want the output flattened.
        """
        def __init__(self, cnns, fcs):
            self.cnns = cnns
            self.fcs = fcs
            self.pl = 0

        def __str__(self):
            return '[cnn:' + str(self.cnns) + '   fcs:' + str(self.fcs) +  '    res_blocks:' + ']'

        def make_pl(self, net, param, scope_prefix=None):
            features = []
            self.pl += 1
            if scope_prefix is None:
                scope_prefix = self.scope_prefix
            for i, line in enumerate(param):
                if line is None:
                    continue
                pl = net
                for j, layer in enumerate(line):
                    if isinstance(layer, int):
                        pl = slim.max_pool2d(pl, layer, scope=scope_prefix + 'pl' + str(self.pl) + 'line' + str(i) + 'pool' + str(j))
                        continue
                    pl = slim.conv2d(pl, layer[0], abs(layer[1]), 
                                                   padding='SAME' if layer[1] > 0 else "VALID",
                                                   scope=scope_prefix + 'pl' + str(self.pl) + 'line' + str(i) + 'conv' + str(j))
                features.append(pl)
            net = tf.concat(features,3, scope_prefix + 'pl' + str(self.pl)+'concat')
            return net

        def make_res(self, net, param):
            scope_prefix = self.scope_prefix
            net = slim.batch_norm(
                net, activation_fn=tf.nn.relu, scope=scope_prefix + 'block' + str(self.block_cnt) + 'preact')
            residual = net
            for layer_i, block in enumerate(param):
                if block[0] is None:
                    residual = self.make_pl(residual, block, scope_prefix + 'block' + str(self.block_cnt))
                    continue
                if layer_i == len(param)-1:
                    residual = slim.conv2d(residual, block[0], abs(block[1]), padding='SAME' if block[1] > 0 else "VALID",
                                        scope=scope_prefix + 'block' + str(self.block_cnt) + 'conv' + str(layer_i), 
                                        normalizer_fn=None, activation_fn=None)
                else:
                    residual = slim.conv2d(residual, block[0], abs(block[1]), 
                                            padding='SAME' if block[1] > 0 else "VALID",
                                            scope=scope_prefix + 'block' + str(self.block_cnt) + 'conv' + str(layer_i))
            # print("net.shape=", net.shape,'   residual.shape= ',residual.shape)
            if net.shape[1:] == residual.shape[1:]:
                # print('shortcut = net')
                shortcut = net
            else:
                # print('shortcut = slim.conv2d')
                # make a convolution with a 1*1 kernel tp keep the dimensions consistent
                shortcut = slim.conv2d(net, int(residual.shape[-1]), 1, padding='SAME', 
                                    normalizer_fn=None, activation_fn=None, 
                                    scope=scope_prefix + 'block' + str(self.block_cnt) + 'shortcut')
            net = tf.add(shortcut, residual, scope_prefix + 'block' + str(self.block_cnt) + 'merge')
            net = tf.nn.relu(net)
            self.block_cnt += 1
            self.need_bn = True
            return net

        def construct(self, input, keep_prob=1.0, scope_prefix=""):
            conv_cnt = 1
            pool_cnt = 1
            self.block_cnt = 1
            net = input
            self.scope_prefix = scope_prefix

            # we need BN after residual networks
            self.need_bn = False
            def try_bn():
                if self.need_bn:
                    self.need_bn = False
                    net = slim.batch_norm(net, activation_fn=tf.nn.relu, 
                                          scope=scope_prefix + 'block' + str(self.block_cnt) + 'postbn')
            for param in self.cnns:
                if isinstance(param, int):
                    try_bn()
                    if param > 0:
                        net = slim.max_pool2d(net, param, scope=scope_prefix + 'pool' + str(pool_cnt))
                    else:
                        if -param == 1:
                            net = math_ops.reduce_mean(net, [1, 2], name=scope_prefix + 'pool' + str(pool_cnt), keepdims=True)
                        else:
                            net = slim.avg_pool2d(net, -param, scope=scope_prefix + 'pool' + str(pool_cnt))
                    pool_cnt += 1
                elif param[0] is None:
                    try_bn()
                    net = self.make_pl(net, param)
                elif not isinstance(param[0], int):
                    net = self.make_res(net, param)
                else:
                    try_bn()
                    net = slim.conv2d(net, param[0], abs(param[1]), scope=scope_prefix+'conv' + str(conv_cnt),
                                      padding='SAME' if param[1] > 0 else "VALID")
                    conv_cnt += 1
            if self.fcs is None:
                return net
            net = slim.flatten(net)
            for idx, param in enumerate(self.fcs):
                net = slim.fully_connected(net, param, scope=scope_prefix+'fc' + str(idx))
                net = slim.dropout(net, keep_prob=keep_prob)
            return net

    def __init__(self, cnn_params=Params([[16,5],2,[32,5],2], [1024]), 
                 kp=0.8, lr_init=0.005, lr_dec_rate=0.95, batch_size=128, data_aug=True,
                 epoch=10, verbose=False, act=tf.nn.relu, l2=1e-8, path=None):
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
        self.ts = {}
        self.data_aug = data_aug
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

    def preprocess(self, x, is_train):
        def augment():
            def flip(img):
                return tf.image.random_flip_left_right(tf.image.random_flip_left_right(img))
            x_trans = tf.map_fn(flip, x)
            return x_trans
        def no_augment():
            return x
        x_trans = tf.cond(is_train, augment, no_augment)
        return x_trans

    def construct_model(self):
        tf.reset_default_graph()
        n_xs, slen = self.X.shape
        slen = int(round(np.sqrt(slen)))
        n_labels = self.Y.shape[1]
        x = tf.placeholder(tf.float32, [None, slen*slen], name='x')
        x_trans = tf.reshape(x, [-1, slen, slen, 1])
        is_train = tf.placeholder(tf.bool, [], name='is_train')
        if self.data_aug:
            x_trans = self.preprocess(x_trans, is_train)
        kp = tf.placeholder(tf.float32, [], name='kp')
        y = tf.placeholder(tf.float32, [None, n_labels], name='y')
        net = x_trans
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=self.act,
                    normalizer_fn=tf.layers.batch_normalization,
                    normalizer_params={'training': is_train, 'momentum': 0.95},
                    weights_regularizer=slim.l2_regularizer(self.l2)):
            with slim.arg_scope([slim.batch_norm], is_training=is_train):
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
        tf.summary.FileWriter(self.path,sess.graph)
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


from nets import CNN_structures
def main_mnist_res():
    data, labels = read_mnist_dl()
    # print(data.shape)
    # data = data[]
    # labels = labels[:5000]
    params1 = CNN.Params(
        [[64,3],2,[[64,3]]*2,[[64,3]]*2,[[64,3]]*2, 2, [[128,3]]*2,[[128,3]]*2,[[128,3]]*2, -1], [512])
    params2 = CNN.Params(
        [ [16,5],2,
           [[None,  
                    [[32,5],[32,5]], 
                    [[32,3],[32,3]]],
           [64,3],[64,3]
            ]],
          [512])
    params3 = CNN.Params([
            [128,3],2,
            [[None,
                    [[64,1]], 
                    [[64,1],[96,3]],
                    [[64,1],[96,3], [128,5]],
                    ],
            ]],
            [512])
    params4 = CNN.Params([
            [128,3],2,
            [[None,
                    [[64,1]], 
                    [[96,3]],
                    [[128,5]],
                    ],
            ]],
            [512])
    params5 = CNN.Params([
            [128,3],2,
            [[None,
                    [[64,1]], 
                    [[96,3]],
                    [[128,5]],
                    ],
            [[128,1]]
            ]],
            [512])

    cnn = CNN(path='log_resCNN',epoch=25, verbose=True, batch_size=128, cnn_params=CNN_structures.zeronn3, data_aug=False)
    cnn.fit(data, labels, 0.1)


if __name__ == "__main__":
    # main_mnist()
    main_mnist_res()


"""
mnist9/1-testacc: 98.8%  10epoch
[[32,3],2,[[32,3]]*2,[[32,3]]*2,[[32,3]]*2, 2, [[64,3]]*2,[[64,3]]*2,[[64,3]]*2], [256]
mnist9/1: train_acc:  0.999982905982906    test_acc: 0.9952307692307693  25epoch
[[64,3],2,[[64,3]]*2,[[64,3]]*2,[[64,3]]*2, 2, [[128,3]]*2,[[128,3]]*2,[[128,3]]*2], [512]
mnist9/1: train_acc:  0.9992991452991453    test_acc: 0.994  25epoch
[[64,3],2,[[64,3]]*2,[[64,3]]*2,[[64,3]]*2, 2, [[128,3]]*2,[[128,3]]*2,[[128,3]]*2, -1], [512]
"""