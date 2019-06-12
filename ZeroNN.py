# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: network model. refer to AlphaGo Zero's network. One way in and two ways out.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import control_flow_ops
from sklearn.metrics import *
from data_utils import *
from global_defs import *
from CNN import CNN
import log
import copy


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


class ZeroNN:
    def __init__(self, common_cnn=CNN.Params([[32,3],[64,3],[128,3]],None), 
                 policy_cnn=CNN.Params([[4,1]], []), value_cnn=CNN.Params([[2,1],2], [128]), 
                 kp=0.5, lr_init=0.05, lr_dec_rate=0.95, batch_size=256, ckpt_idx=-2,
                 epoch=10, verbose=None, act=tf.nn.relu, l2=1e-7, path=None, lock_model_path=None):
        """
        verbose: set verbose an integer to output the training history or None not to output
        """
        self.common_cnn = common_cnn
        self.policy_cnn = policy_cnn
        self.value_cnn = value_cnn
        self.kp = kp
        self.lr_init = lr_init
        self.lr_dec_rate = lr_dec_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.lock_model_path = lock_model_path
        self.act = act
        self.l2 = l2
        self.path = None if path is None else mkdir(path)
        self.logger = log.Logger(None if self.path is None else join(self.path, logfn('ZeroNN-' + curr_time_str())), 
                                 verbose is not None)
        self.sess = None
        self.ts = {}
        self.ckpt_idx = ckpt_idx
        self.var_names = ['x','kp', 'y_value','y_policy', 'is_train', 
        'loss_policy', 'loss_value', 'pred_value','acc_value', 
        'pred_policy', 'loss_l2', 'loss_total', 'global_step','train_step']
        self.trained_model_paths = []

    def print_vars(self, graph=None, vars=True, ops=True):
        if graph is None:
            graph = tf.get_default_graph()
        variable_names = tf.global_variables()
        for name in variable_names:
            print(name)
        op = graph.get_operations()
        for i in op:
            print(i)

    def init_vars(self):
        for ts in self.var_names:
                self.ts[ts] = tf.get_collection(ts)[0]

    def __str__(self):
        return "ZeroNN-- common_cnns: {} \tfcs: {} \tkp: {} \tlr_init: {} \tlr_dec_rate: {} \tbatch_size: {} \tepoch: {} \tact: {}".format(
            self.common_cnns, self.fcs, self.kp, self.lr_init, self.lr_dec_rate, self.batch_size, self.epoch, str(self.act).split(' ')[1] if self.act is not None else 'NONE')

    def init_training_data(self, X, Y_policy, Y_value, reserve_test):
        if reserve_test is not None:
            xy_tr, xy_te = labeled_data_split([X, Y_policy, Y_value], 1.0 - reserve_test)
            X, Y_policy, Y_value = xy_tr
            X_te, Y_policy_te, Y_value_te = xy_te
            self.X_te = X_te
            self.Y_policy_te =  Y_policy_te
            # self.Y_policy_te = labels2one_hot(Y_policy_te)
            self.Y_value_te = Y_value_te
        else:
            self.X_te = None
            self.Y_policy_te = None
            self.Y_value_te = None
        self.X = X
        self.Y_policy = Y_policy
        # self.Y_policy = labels2one_hot(Y_policy)
        self.Y_value = Y_value

    def fit(self, X, Y_policy, Y_value, reserve_test=None, refresh_saving=False):
        """
        If you wanna extract test set automatically, set reserve_test the ratio for test set.
        X should be a n*rows*cols*channels, where channels is number of histories.
        """
        self.init_training_data(X, Y_policy, Y_value, reserve_test)
        self.construct_model()
        self.init_hists(refresh_saving)
        self.init_sess(refresh_saving)
        self.train()

    def construct_model(self):
        tf.reset_default_graph()
        
        n_xs, rows, cols, channels = self.X.shape
        n_labels_value = self.Y_value.shape[1]
        n_labels_policy = self.Y_policy.shape[1]
        x = tf.placeholder(tf.float32, [None, rows, cols, channels], name='x')
        x_trans = tf.image.flip_up_down(tf.image.flip_left_right(x))
        kp = tf.placeholder(tf.float32, [], name='kp')
        y_value = tf.placeholder(tf.int8, [None, n_labels_value], name='y_value')
        y_policy = tf.placeholder(tf.float32, [None, n_labels_policy], name='y_policy')
        is_train = tf.placeholder(tf.bool, [], name='is_train')

        # construct shared, policy and value networks: one way in, two ways out
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=self.act,
                    normalizer_fn=tf.layers.batch_normalization,
                    normalizer_params={'training': is_train, 'momentum': 0.95},
                    weights_regularizer=slim.l2_regularizer(self.l2)):
            output_common = self.common_cnn.construct(x_trans, kp, "common_")
            output_policy = self.policy_cnn.construct(output_common, kp, "policy_")
            logits_policy = slim.fully_connected(output_policy, rows * cols, activation_fn=tf.nn.softmax, scope='logits_policy')
            output_value = self.value_cnn.construct(output_common, kp, "value_")
            logits_value = slim.fully_connected(output_value, 1, activation_fn=tf.nn.tanh, scope='logits_value')

        # define loss and accuracies of value network
        # actually value network return a scalar
        loss_value = tf.losses.mean_squared_error(labels=y_value, predictions=logits_value, scope='loss_value')
        # pred_value is actually the same as logits_value, we are using an alias
        pred_value = tf.add_n([logits_value], name='pred_value')
        # print(logits_value)
        corrects_value = tf.equal(tf.cast(tf.round(pred_value), tf.int8),y_value)
        acc_value = tf.reduce_mean(tf.cast(corrects_value, tf.float32),name='acc_value')

        # define loss and accuracies of policy network
        # tf.losses.softmax_cross_entropy() cannot be used for cross_entropy calculation since the labels are not ONE-HOT
        loss_policy = tf.reduce_mean(-y_policy*tf.log(tf.clip_by_value(logits_policy,1e-10,1.0)), name='loss_policy')
        pred_policy = tf.add_n([logits_policy], name='pred_policy')
        
        # regularization_loss = tf.add_n(tf.losses.get_regularization_losses, name='loss_l2') 
        loss_l2 = tf.losses.get_regularization_loss(scope='loss_l2')
        loss_total = tf.add_n([loss_policy, loss_value, loss_l2], name='loss_total')
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)

        # learning rate annealing
        lr = tf.train.exponential_decay(
            self.lr_init,
            global_step,
            n_xs / self.batch_size, self.lr_dec_rate,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = slim.learning.create_train_op(
            loss_total,  optimizer, global_step=global_step)
        
        # ensure batch normalization is done before forwarding
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_step = control_flow_ops.with_dependencies([updates], train_step, name='train_step')
        locs = locals()
        for var in self.var_names:
            tf.add_to_collection(var, locs[var])

    def next_batch(self):
        batch_sz = self.batch_size
        indices = list(range(self.curr_tr_batch_idx, self.curr_tr_batch_idx+batch_sz))
        self.curr_tr_batch_idx = (batch_sz + self.curr_tr_batch_idx) % self.X.shape[0]
        indices = [i%self.X.shape[0] for i in indices]
        return [self.X[indices], self.Y_policy[indices], self.Y_value[indices]]

    def run_eval(self, X, Y_policy, Y_value):
        loss_policy_sum = 0.0
        loss_value_sum = 0.0
        loss_total_sum = 0.0
        acc_value_sum = 0.0
        correct_preds_value = 0.0
        feed_dict = {self.ts['kp']: 1.0, self.ts['is_train']: False}
        for batch_idx in range(0,X.shape[0],self.batch_size):
            batch_idx_next = min(X.shape[0], batch_idx + self.batch_size)
            batch_xs = X[batch_idx:batch_idx_next]
            batch_ys_policy = Y_policy[batch_idx:batch_idx_next]
            batch_ys_value = Y_value[batch_idx:batch_idx_next]
            feed_dict.update({self.ts['x']: batch_xs, self.ts['y_value']: batch_ys_value, self.ts['y_policy']: batch_ys_policy})
            [loss_policy, loss_value, loss_total, acc_value] = self.sess.run(
                [self.ts['loss_policy'], self.ts['loss_value'], self.ts['loss_total'], self.ts['acc_value']],feed_dict=feed_dict)
            mult = batch_idx_next - batch_idx
            loss_policy_sum += loss_policy * mult
            loss_value_sum += loss_value * mult
            loss_total_sum += loss_total * mult
            acc_value_sum += acc_value * mult
        return [loss_policy_sum / X.shape[0], 
                loss_value_sum / X.shape[0], 
                loss_total_sum / X.shape[0], 
                acc_value_sum / X.shape[0]]

    def init_sess(self, refresh_saving):
        """
        return whether use new parameters
        """
        path_meta = join(self.path, '0.meta')
        if self.path is not None and exists(path_meta):
            tf.reset_default_graph()
            sess = tf.Session()
            self.saver = tf.train.import_meta_graph(path_meta)
            self.logger.log("Find the meta in file", self.path)
        else:
            self.logger.log("Init new meta")
            self.saver = tf.train.Saver(max_to_keep=1000)
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
        self.init_vars()
        self.sess = sess
        if not refresh_saving and self.path is not None: 
            try: 
                if isinstance(self.ckpt_idx, str):
                    self.saver.restore(sess,self.ckpt_idx) 
                    self.logger.log("Find the check point ",self.ckpt_idx," in file", self.path)
                    return True
                
                ckpt = tf.train.get_checkpoint_state(self.path)
                if ckpt and ckpt.all_model_checkpoint_paths:
                    # print(ckpt.all_model_checkpoint_paths[self.ckpt_idx])
                    self.ckpt_path = ckpt.all_model_checkpoint_paths[self.ckpt_idx]
                    self.saver.restore(sess,ckpt.all_model_checkpoint_paths[self.ckpt_idx]) 
                    self.logger.log("Find the check point ",self.ckpt_idx," in file", self.path)
                    return True
            except: 
                self.logger.log("Init new parameters")
                return False

    def train(self):
        sess = self.sess
        self.curr_tr_batch_idx = 0
        it_pep = round(self.X.shape[0] / self.batch_size)
        if self.path is not None:
            self.saver.save(sess, self.path + '/0', write_meta_graph=True)
        it_epoch = 0
        x_t = self.ts['x']
        kp_t = self.ts['kp']
        y_value_t = self.ts['y_value']
        y_policy_t = self.ts['y_policy']
        is_train_t = self.ts['is_train']
        train_step_t = self.ts['train_step']
        global_step_t = self.ts['global_step']
        for i in range(round(self.epoch * self.X.shape[0] / self.batch_size)+1):
            batch_xs, batch_ys_policy, batch_ys_value = self.next_batch()
            feed_dict = {x_t: batch_xs, kp_t: self.kp, y_value_t: batch_ys_value, y_policy_t: batch_ys_policy, is_train_t: True}
            sess.run(train_step_t, feed_dict=feed_dict)
            # global_step = sess.run(global_step_t, feed_dict=feed_dict)
            if i % it_pep == 0:
                global_step = sess.run(global_step_t, feed_dict=feed_dict)
                train_eval = self.run_eval(self.X, self.Y_policy, self.Y_value)
                test_eval = [-1.0 for i in range(4)] if self.X_te is None \
                      else self.run_eval(self.X_te, self.Y_policy_te, self.Y_value_te)
                self.train_hists.append([global_step//it_pep] + train_eval)
                self.test_hists.append([global_step//it_pep] + test_eval)
                it_epoch += 1
                if self.verbose is not None and it_epoch % self.verbose == 0:
                    self.logger.log('\nglobal_step:',global_step, '  epoch:',global_step//it_pep,
                          '\n   items:        [loss_policy,       loss_value,           loss_total,            acc_value]:',
                          '\n   train_eval: ',train_eval, 
                          '\n   test_eval:',  test_eval)
                if self.path is not None:
                    path = self.saver.save(sess, self.path + '/model.ckpt', global_step=global_step_t, write_meta_graph=False)
                    # print(path)
                    if self.lock_model_path is not None:
                        lock_model_path.acquire()
                    self.trained_model_paths.append(path)
                    if self.lock_model_path is not None:
                        lock_model_path.release()
                    self.save_hists()
        #if self.path is not None:
        #    self.saver.save(sess, self.path + '/0', write_meta_graph=True)

    def init_hists(self, refresh_saving):
        if refresh_saving:
            self.train_hists = []
            self.test_hists = []
            return None
        path_train = join(self.path, npfn('train'))
        path_test = join(self.path, npfn('test'))
        self.train_hists = [] if not exists(path_train) else np.load(path_train).tolist()
        self.test_hists = [] if not exists(path_test) else np.load(path_test).tolist()

    def save_hists(self):
        if not exists(self.path):
            return
        path_train = join(self.path, npfn('train'))
        path_test = join(self.path, npfn('test'))
        np.save(join(self.path, npfn('train')), np.array(self.train_hists))
        np.save(join(self.path, npfn('test')), np.array(self.test_hists))

    def predict(self, X):
        if self.sess is None:
            if not self.init_sess(refresh_saving=False):
                raise Exception("Error: trying to predict without trained network")
        pred_value, pred_policy = self.sess.run([self.ts['pred_value'], self.ts['pred_policy']], 
                             feed_dict={self.ts['x']: X, self.ts['kp']: 1.0, self.ts['is_train']: False})
        return [pred_value, pred_policy]


def main_sim_train():
    num_samples = 5000
    rows = 5
    cols = 5
    channel = 4
    X = np.random.rand(num_samples,rows,cols, channel)
    Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
    Y_policy = np.random.rand(num_samples,rows*cols)
    clf = ZeroNN(verbose=2, path='ZeroNN', ckpt_idx='ZeroNN/model.ckpt-309')
    clf.fit(X, Y_policy, Y_value, 0.1)
    print(X[:2].shape)
    pred_value, pred_policy = clf.predict(X[:2])
    print(pred_value, pred_policy)


def _test_flip():
    data = np.random.randint(0,2,[2,3,3,2], dtype=np.int8)
    x = tf.convert_to_tensor(data)
    print(data[0][:,:,0])
    print(data[1][:,:,1])
    with tf.Session() as sess:
        flipped0 = tf.image.flip_up_down(x)
        data = flipped0.eval()
        print(data[0][:,:,0])
        print(data[1][:,:,1])

if __name__=='__main__':
    main_sim_train()




