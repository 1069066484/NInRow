# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: network model. refer to AlphaGo Zero's network. One way in and two ways out.
"""
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.python.ops import control_flow_ops
from sklearn.metrics import *
from utils.data import *
from global_defs import *
from nets.CNN import CNN
from utils import log
import copy
from scipy import misc
from tensorflow.contrib.slim import nets
from nets import CNN_structures
import matplotlib.pyplot as plt 
from functools import reduce
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ZeroNN:
    def __init__(self, 
                 common_cnn=CNN_structures.zeronn8, 
                 policy_cnn=CNN.Params([[2,1]], []), 
                 value_cnn=CNN.Params([[1,1],1], [256]), 
                 kp=0.5, lr_init=2e-6, lr_dec_rate=0.9999, batch_size=256, ckpt_idx=-1, save_epochs=2,
                 epoch=10, verbose=None, act=tf.nn.relu, l2=1e-4, path=None, lock_model_path=None, init_check=True, 
                 num_samples=None, logger=None, init_path=None, use_prek=True, save_best=False, trained_model_paths=[]):
        """
        verbose:    set verbose an integer to output the training history or None not to output
        logger:     if a logger is provided, the outputs will use the given logger
        """
        self.common_cnn = common_cnn
        self.policy_cnn = policy_cnn
        self.value_cnn = value_cnn
        self.kp = kp
        self.num_samples = num_samples
        self.lr_init = lr_init
        self.lr_dec_rate = lr_dec_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.lock_model_path = lock_model_path
        self.act = act
        self.save_epochs = save_epochs
        self.l2 = l2
        self.init_check = init_check
        self.path = None if path is None else mkdir(path)
        self.path_sum = None if path is None else mkdir(join(self.path, 'summaries'))
        self.logger = log.Logger(None if self.path is None else join(self.path, logfn('ZeroNN-' + curr_time_str())), 
                                 verbose is not None)
        if logger is not None:
            self.logger = logger
        self.sess = None
        self.ts = {}
        self.ckpt_idx = ckpt_idx

        # collections of variables
        self.var_names = ['x','kp', 'y_value','y_policy', 'is_train', 'acc_policy',
        'loss_policy', 'loss_value', 'pred_value','acc_value', 'lr', 'merged_sum',
        'pred_policy', 'loss_l2', 'loss_total', 'global_step','train_step']
        self.trained_model_paths = trained_model_paths
        self.train_hists = []
        self.test_hists = []
        self.init_path = init_path
        self.use_prek = use_prek
        self.save_best = save_best
        self.best_test_loss = 10000

    def predict_avail(self):
        return len(self.ts) != 0

    def print_vars(self, graph=None, vars=True, ops=True):
        """
        print graph variables(for debugging only)
        """
        if graph is None:
            graph = tf.get_default_graph()
        variable_names = tf.global_variables()
        for name in variable_names:
            print(name)
        op = graph.get_operations()
        for i in op:
            print(i)

    def init_vars(self):
        """
        Get collections in the graph and save the necessary variables into the dictionary
        """
        for ts in self.var_names:
            try:
                self.ts[ts] = tf.get_collection(ts)[0]
            except:
                print(ts, 'collection failed')

    def __str__(self):
        return "ZeroNN"
        return "ZeroNN-- common_cnn: {} \tfcs: {} \tkp: {} \tlr_init: {} \tlr_dec_rate: {} \tbatch_size: {} \tepoch: {} \tact: {}".format(
            self.common_cnn, self.fcs, self.kp, self.lr_init, self.lr_dec_rate, self.batch_size, self.epoch, str(self.act).split(' ')[1] if self.act is not None else 'NONE')

    def copy_possible_params(self):
        # NinRow
        
        if self.init_path is None:
            return
        print('copy possible params from', self.init_path, '...')
        vars = tf.global_variables()
        reader = tf.train.NewCheckpointReader(self.init_path)
        var_to_shape_map = reader.get_variable_to_shape_map() 
        init_cnt = 0
        for var in vars:
            name = var.name.split(':')[0]
            if name in var_to_shape_map and name.count('Adam') == 0:
                var_ref = reader.get_tensor(name)
                # print(var, var_ref)
                if var_ref.shape == var.shape:
                    # print(var.name)
                    self.sess.run(tf.assign(var, var_ref))
                    init_cnt += 1
        print(init_cnt, 'params copied')
        if self.path is not None:
            self.saver.save(self.sess, self.path + '/0', write_meta_graph=True)
            print('params written')

    def init_training_data(self, X, Y_policy, Y_value, reserve_test):
        """
        Training and test data are divided here if required.
        """
        if reserve_test is not None:
            xy_tr, xy_te = labeled_data_split([X, Y_policy, Y_value], 1.0 - reserve_test)
            X, Y_policy, Y_value = xy_tr
            X_te, Y_policy_te, Y_value_te = xy_te
            self.X_te = X_te
            self.Y_policy_te =  Y_policy_te
            self.Y_value_te = Y_value_te
        else:
            self.X_te = None
            self.Y_policy_te = None
            self.Y_value_te = None
        self.X = X
        self.Y_policy = Y_policy
        self.Y_value = Y_value

    def fit(self, X, Y_policy=None, Y_value=None, reserve_test=None, refresh_saving=False):
        """
        If you wanna extract test set automatically, set reserve_test the ratio for test set.
        X should be a n*rows*cols*channels, where channels is number of histories.
        """
        if Y_policy is None or Y_value is None:
            X, Y_policy, Y_value = X[0], X[1], X[2]
        self.init_training_data(X, Y_policy, Y_value, reserve_test)
        if self.sess is None and not refresh_saving:
            self.construct_model()
            self.init_hists(refresh_saving)
            self.init_sess(refresh_saving)
        self.train()

    def tf_random_rotate90(self, image, rotate_prob=0.5):
        rotated = tf.image.rot90(image)
        rand = tf.random_uniform([], minval=0, maxval=1)
        return tf.cond(tf.greater(rand, rotate_prob), lambda: image, lambda: rotated)
        
    def preprocess(self, x, y_policy, is_train, rows, cols):
        """
        Fixed convolution and data augmentation are performed here.
        Four channels of input x:
            1. pieces of previous player
            2. pieces of the current player
            3. the node's movement(last movement)
            4. the current player
        The board can be reversed in terms of the opponent's view
        """
        # The codes below do convolution using fixed kernels
        # weights = [0.1,0.2,0.3,0.2,0.1]
        if self.use_prek:
            weights = [0.1,0.2,0.4,0.2,0.1]
            len_w = len(weights)
            tl2br = np.diag(weights)
            filter_tl2br = tf.constant_initializer(tl2br.reshape([len_w,len_w,1,1]))
            filter_tr2bl = tf.constant_initializer(np.rot90(tl2br).reshape([len_w,len_w,1,1]))
            l2r = np.array([0 for _ in range(len_w)] * (len_w//2) + weights + [0 for _ in range(len_w)] * (len_w//2)).reshape([len_w,len_w])
            filter_l2r = tf.constant_initializer(l2r.reshape([len_w,len_w,1,1]))
            filter_t2b = tf.constant_initializer(np.rot90(l2r).reshape([len_w,len_w,1,1]))
        
            x = tf.concat([x] + 
                          [slim.conv2d(x[:,:,:,(ch-1):ch], 1, len_w, 
                                       weights_initializer=init,trainable=False)
                           for ch in [1,2] for init in 
                           [filter_tl2br, filter_tr2bl, filter_l2r, filter_t2b]], 
                          axis=3)

        lastdim = int(x.shape[-1])

        # the codes below performs data augmentation
        y_policy_trans = tf.reshape(y_policy, [-1, rows, cols, 1])
        # x and y_policy should be stacked, it is not right to rotate x only!
        x_y_policy = tf.concat([x, y_policy_trans], axis=3)
        x_y_policy = self.tf_random_rotate90(x_y_policy)
        def flip(img):
            return tf.image.random_flip_left_right(tf.image.random_flip_up_down(img))
        x_y_policy = tf.map_fn(flip, x_y_policy)
        x_trans = x_y_policy[:,:,:,:lastdim]
        y_policy_trans = x_y_policy[:,:,:,lastdim]
        y_policy_trans = tf.reshape(y_policy_trans, [-1, rows * cols])

        # lambda: image, lambda: rotated
        x_ret, y_policy_ret = tf.cond(is_train, lambda: [x_trans, y_policy_trans], lambda: [x, y_policy])
        return x_ret, y_policy_ret

    def construct_model(self):
        """
        Construct the network. We make use of CNN.Params for help, which makes the codes clean and clear.
        """
        tf.reset_default_graph()
        n_xs, rows, cols, channels = self.X.shape
        n_labels_value = self.Y_value.shape[1]
        n_labels_policy = self.Y_policy.shape[1]
        x = tf.placeholder(tf.float32, [None, rows, cols, channels], name='x')
        y_value = tf.placeholder(tf.int8, [None, n_labels_value], name='y_value')
        y_policy = tf.placeholder(tf.float32, [None, n_labels_policy], name='y_policy')
        kp = tf.placeholder(tf.float32, [], name='kp')
        is_train = tf.placeholder(tf.bool, [], name='is_train')
        lr = tf.placeholder(tf.float32, [], name='lr')
        x_trans, y_policy_trans = self.preprocess(x, y_policy, is_train, rows, cols)
        self.x_trans = x_trans
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
        corrects_value = tf.equal(tf.cast(tf.round(pred_value), tf.int8),y_value)
        acc_value = tf.reduce_mean(tf.cast(corrects_value, tf.float32),name='acc_value')

        # define loss and accuracies of policy network
        # tf.losses.softmax_cross_entropy() cannot be used for cross_entropy calculation since the labels are not ONE-HOT
        cross_entropy_policy = -tf.reduce_sum(y_policy_trans*tf.log(tf.clip_by_value(logits_policy,1e-10,1.0)),1)
        loss_policy = tf.reduce_mean(cross_entropy_policy, name='loss_policy')
        pred_policy = tf.add_n([logits_policy], name='pred_policy')
        corrects_policy = tf.equal(tf.cast(tf.argmax(pred_policy), tf.int8),tf.cast(tf.argmax(y_policy), tf.int8))
        acc_policy = tf.reduce_mean(tf.cast(corrects_policy, tf.float32),name='acc_policy')
        # pred_policy = tf.nn.softmax(logits_policy, name='pred_policy')

        # regularization_loss = tf.add_n(tf.losses.get_regularization_losses, name='loss_l2') 
        loss_l2 = tf.losses.get_regularization_loss(name='loss_l2')
        loss_total = tf.add_n([loss_policy, loss_value, loss_l2], name='loss_total')
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)

        # learning rate annealing
        '''
        lr = tf.train.exponential_decay(
            self.lr_init,
            global_step,
            (n_xs if self.num_samples is None else self.num_samples) / self.batch_size, 
            self.lr_dec_rate,
            staircase=True)
        lr = tf.reduce_max([lr, 1e-5], name='lr')
        lr = tf.Variable(1e-4, 'l2')
        '''
 
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = slim.learning.create_train_op(
            loss_total,  optimizer, global_step=global_step)
        
        # ensure batch normalization is done before forwarding
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_step = control_flow_ops.with_dependencies([updates], train_step, name='train_step')
        
        # unused for now
        tf.summary.scalar("lr", lr)
        tf.summary.scalar("loss_total", loss_total)
        tf.summary.scalar("loss_l2", loss_l2)
        tf.summary.scalar("loss_policy", loss_policy)
        tf.summary.scalar("loss_value", loss_value)
        tf.summary.scalar("acc_value", acc_value)
        merged_sum = tf.summary.merge_all(name='merged_sum')
        
        # we use a trick to collect the variables 
        locs = locals()
        for var in self.var_names:
            tf.add_to_collection(var, locs[var])

    def next_batch(self):
        """
        get next training batch
        """
        batch_sz = self.batch_size
        indices = list(range(self.curr_tr_batch_idx, self.curr_tr_batch_idx+batch_sz))
        self.curr_tr_batch_idx = (batch_sz + self.curr_tr_batch_idx) % self.X.shape[0]
        indices = [i%self.X.shape[0] for i in indices]
        return [self.X[indices], self.Y_policy[indices], self.Y_value[indices]]

    def run_eval(self, X, Y_policy=None, Y_value=None):
        """
        run evaluation on the given data
        Evaluations are performed in batches and the batch results are synthesized
        """
        self.check_init()
        if Y_policy is None or Y_value is None:
            X, Y_policy, Y_value = X[0], X[1], X[2]
        sums = [0.0 for _ in range(5)]
        feed_dict = {self.ts['kp']: 1.0, self.ts['is_train']: False, self.ts['lr']: 1e-5}
        for batch_idx in range(0,X.shape[0],self.batch_size):
            # acuiqre the test batches
            batch_idx_next = min(X.shape[0], batch_idx + self.batch_size)
            batch_xs = X[batch_idx:batch_idx_next]
            batch_ys_policy = Y_policy[batch_idx:batch_idx_next]
            batch_ys_value = Y_value[batch_idx:batch_idx_next]
            feed_dict.update({self.ts['x']: batch_xs, 
                              self.ts['y_value']: batch_ys_value, 
                              self.ts['y_policy']: batch_ys_policy})
            # run the evaluation for the indicated batch
            vals = self.sess.run(
                [self.ts['loss_policy'], 
                 self.ts['loss_value'], 
                 self.ts['loss_total'], 
                 self.ts['acc_value'],
                 self.ts['acc_policy']],
                 feed_dict=feed_dict)
            mult = batch_idx_next - batch_idx
            for i in range(len(sums)):
                sums[i] += vals[i] * mult
            # summarize the evaluation
        return [round(s / X.shape[0], 10) for s in sums]

    def init_sess(self, refresh_saving):
        """
        return whether use new parameters
        """
        path_meta = join(self.path, '0.meta')
        gpu_options = tf.GPUOptions(allow_growth=True)
        if self.path is not None and exists(path_meta):
            tf.reset_default_graph()
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.saver = tf.train.import_meta_graph(path_meta)
            self.logger.log("Find the meta in file", self.path)
        else:
            self.logger.log("Init new meta")
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.saver = tf.train.Saver(max_to_keep=1000)
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
                pass
            self.logger.log("Init new parameters")
            self.copy_possible_params()
            return False
        print('return', not refresh_saving, self.path is not None, 
              not refresh_saving and self.path is not None)
        return False

    def train(self):
        sess = self.sess
        self.curr_tr_batch_idx = 0
        it_pep = max(round(self.X.shape[0] / self.batch_size), 1)
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
        lr = self.ts['lr']
        writer = tf.summary.FileWriter(self.path_sum, sess.graph)

        for i in range(round(self.epoch * self.X.shape[0] / self.batch_size)+1):
            lr_val = self.lr_init
            batch_xs, batch_ys_policy, batch_ys_value = self.next_batch()
            feed_dict = {x_t: batch_xs, kp_t: self.kp, y_value_t: batch_ys_value, 
                         y_policy_t: batch_ys_policy, is_train_t: True, lr:lr_val}
            sess.run(train_step_t, feed_dict=feed_dict)
            print(sess.run(self.x_trans, feed_dict=feed_dict)[0,:,:,0])
            # self.logger.log('learning rate:', sess.run(self.ts['lr']))
            """
            tf.summary.scalar("loss_total", loss_total)
            tf.summary.scalar("loss_l2", loss_l2)
            tf.summary.scalar("loss_policy", loss_policy)
            tf.summary.scalar("loss_value", loss_value)
            """
            # print(sess.run([self.ts["loss_total"], self.ts["loss_l2"], self.ts["loss_policy"], self.ts["loss_value"],], feed_dict=feed_dict))
            # for each epoch, refresh logs and models if necessary
            if i % it_pep == 0:
                it_epoch += 1
                if not (it_epoch % self.save_epochs == 0 or it_epoch % self.verbose == 0):
                    continue
                global_step  = sess.run(global_step_t, feed_dict=feed_dict)
                train_eval = self.run_eval(self.X, self.Y_policy, self.Y_value)
                test_eval = [-1.0 for i in range(4)] if self.X_te is None \
                      else self.run_eval(self.X_te, self.Y_policy_te, self.Y_value_te)
                need_save = not self.save_best or self.best_test_loss > test_eval[2]
                if self.best_test_loss > test_eval[2] and self.save_best:
                    self.best_test_loss = test_eval[2]
                self.train_hists.append([global_step] + train_eval)
                self.test_hists.append([global_step] + test_eval)
                if self.verbose is not None and it_epoch % self.verbose == 0:
                    self.logger.log('\nglobal_step:',global_step, '  epoch:',global_step//it_pep,
                          '\n   items:        [loss_policy,  loss_value,   loss_total,    acc_value,   acc_policy]:',
                          '\n   train_eval: ',train_eval)
                    if self.X_te is not None:
                          self.logger.log('   test_eval:  ',  test_eval)
                    if self.init_check:
                        np.set_printoptions(3, suppress=True)
                        # print('Init board:','\n',self.predict(np.zeros([1] + list(self.X.shape[1:]) ) )[1].reshape(self.X.shape[1:3]))
                    
                if self.path is not None and it_epoch % self.save_epochs == 0 and need_save:
                    path = self.saver.save(sess, self.path + '/model.ckpt', global_step=global_step_t, write_meta_graph=False)
                    if self.verbose  is not None:
                        self.logger.log('model saved in',path)
                    if self.lock_model_path is not None:
                        with self.lock_model_path:
                            self.trained_model_paths.append(path)
                    else:
                         self.trained_model_paths.append(path)
                self.save_hists()

    def init_hists(self, refresh_saving=False):
        """
        Initialize training histories:
            If no previous histories are found, create new ones,
            else load the existing histories and append new histories to them
        """
        if self.train_hists is not None and self.test_hists is not None and len(self.train_hists) > 0 and len(self.test_hists) > 0:
            self.best_test_loss = self.test_hists[-1][2]
            return
        if refresh_saving:
            self.train_hists = []
            self.test_hists = []
            self.best_test_loss = 10000000
            return None
        path_train = join(self.path, npfn('train'))
        path_test = join(self.path, npfn('test'))
        # print(path_test)
        try:
            self.train_hists = [] if not exists(path_train) else np.load(path_train).tolist()
            self.test_hists = [] if not exists(path_test) else np.load(path_test).tolist()

        except:
            self.train_hists = []
            self.test_hists = []
        self.best_test_loss = 10000000 if len(self.test_hists) == 0 else self.test_hists[-1][2] 

    def save_hists(self):
        """
        Save training histories
        """
        if not exists(self.path):
            return
        path_train = join(self.path, npfn('train'))
        path_test = join(self.path, npfn('test'))
        np.save(join(self.path, npfn('train')), np.array(self.train_hists))
        np.save(join(self.path, npfn('test')), np.array(self.test_hists))

    def check_init(self):
        if self.sess is None:
            if not self.init_sess(refresh_saving=False):
                raise Exception("Error: trying to predict without trained network")

    def predict(self, X):
        """
        Make prediction using a trained model: ensure a valid model path is provided.
        Output is the policy and win chance of the player identified by the channel3
            
        """
        self.check_init()
        # y_policy is not necessary for prediction
        # but we just needed for placeholder
        pred_value, pred_policy = self.sess.run(
            [self.ts['pred_value'], 
             self.ts['pred_policy']], 
            feed_dict={self.ts['x']: X, 
                       self.ts['kp']: 1.0,
                       self.ts['is_train']: False,
                       self.ts['y_policy']: np.zeros([X.shape[0], X.shape[1]*X.shape[2]])
                                        })
        return [pred_value, pred_policy]

    def sum_plot(self):
        # print("sum_plot")
        self.init_hists(False)
        # [loss_policy,       loss_value,           loss_total,            acc_value]
        loss_div = 10
        train_hists = np.array(self.train_hists).T
        test_hists = np.array(self.test_hists).T
        train_hists[1] /= loss_div
        train_hists[3] /= loss_div
        test_hists[1] /= loss_div 
        test_hists[3] /= loss_div
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        lines = []
        np.set_printoptions(precision=4)
        step = range(train_hists.shape[1])
        lines.append(ax1.plot(step, train_hists[1], '-', linewidth=1, color='b', 
                              label='training loss_policy/' + str(loss_div)))
        lines.append(ax1.plot(step, train_hists[2], '-', linewidth=1, color='r', label='training loss_value'))
        lines.append(ax1.plot(step, train_hists[3], '-', linewidth=1, color='k', 
                              label='training loss_total/' + str(loss_div)))
        lines.append(ax1.plot(step, test_hists[1], '--', linewidth=1, color='b', 
                              label='test loss_policy/' + str(loss_div)))
        lines.append(ax1.plot(step, test_hists[2], '--', linewidth=1, color='r', label='test loss_value'))
        lines.append(ax1.plot(step, test_hists[3], '--', linewidth=1, color='k', 
                              label='test loss_total/' + str(loss_div)))
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title('training history of zeronn')
        ax2 = ax1.twinx()
        lines.append(ax2.plot(step, train_hists[4], '-', linewidth=1, color='g', label='training acc_value'))
        lines.append(ax2.plot(step, test_hists[4], '--', linewidth=1, color='g', label='test acc_value'))
        lines = reduce(lambda i, sum:i+sum, lines)
        ax1.legend(lines, [l.get_label() for l in lines] #, loc='center right'
                   )
        ax2.set_ylabel("value accuracy")
        if self.path is not None:
            plt.savefig(join(self.path_sum, 'hist.png') )


def main_sim_train():
    num_samples = 1024
    rows = 5
    cols = 5
    channel = 4
    X = np.random.rand(num_samples,rows,cols, channel)
    Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
    Y_policy = np.random.rand(num_samples,rows*cols)
    clf1 = ZeroNN(verbose=2, path='ZeroNN_test', batch_size=512)
    clf2 = ZeroNN(verbose=2, path='ZeroNN_test', batch_size=512)
    clf1.fit(X, Y_policy, Y_value, 0.1)
    print(X[:2].shape)
    clf2.fit(X, Y_policy, Y_value, 0.1)
    clf1.fit(X, Y_policy, Y_value, 0.1)
    clf2.fit(X, Y_policy, Y_value, 0.1)
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


def test_copy_possible_params():
    folder_from = r'F:\Software\vspro\NInRow\NInRow\new_train_zeronn\zeronn8_curr\model.ckpt-44838'
    top_folder = mkdir('trans_params_test')
    zeronn = ZeroNN(verbose=10, epoch=1, path=join(top_folder, 'zeronn8'), 
                    save_epochs=10, batch_size=768, init_path=folder_from)

    num_samples = 1
    rows = 11
    cols = 11
    channel = 4
    X = np.random.rand(num_samples,rows,cols, channel)
    Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
    Y_policy = np.random.rand(num_samples,rows*cols)

    zeronn.fit(X, Y_policy, Y_value, 0.1)
    # zeronn.copy_possible_params(folder_from)


def train_comp_kern(which):
    print('use_prekern=',which==1)
    folder = mkdir('kern_comp')
    data = [np.load(join('pure_tr_nn', 'data', npfn('selfplay'+str(i)))) for i in range(3)]
    zeronn = ZeroNN(verbose=10, epoch=500, path=mkdir(join(folder, 'k_'+str(which==1))), save_best=True,
                        save_epochs=20, batch_size=1024, lr_dec_rate=0.9999, use_prek=which==1)
    zeronn.fit(data[0], data[1], data[2], 0.1)


def train_init_net():
    for prek in [True, False]:
        print('prek=',prek)
        folder = mkdir('data_init/net_prek_' + str(prek))
        data = [np.load(join('data_init', npfn('selfplay'+str(i)))) for i in range(3)]
        print('data=',data[0].shape, data[1].shape, data[2].shape)
        zeronn = ZeroNN(verbose=5, epoch=120, path=folder,
                            save_epochs=5, batch_size=2048, use_prek=prek)
        zeronn.fit(data[0], data[1], data[2], 0.1)
    
    
def train_init_net2():
    folder = mkdir('data_init/net')
    data = [np.load(join('data_init', npfn('selfplay'+str(i)))) for i in range(3)]
    print('data=',data[0].shape, data[1].shape, data[2].shape)
    folder = 'zero_nns115_0/NNs'
    zeronn = ZeroNN(verbose=10, epoch=120, path=folder,
                            save_epochs=10, batch_size=2048)
    zeronn.fit(data[0], data[1], data[2], 0.1)


def test_train():
    folder_data  = r'F:\Software\vspro\NInRow\NInRow\zero_nns115\godd_mod\105633_1'
    data = [np.load(join(folder_data, npfn('selfplay'+str(i)))) for i in range(3)]
    data = np.load(r'F:\Software\vspro\NInRow\NInRow\zero_nns115\new_sp\selfplay106343.npz')
    data = [data['sp0'], data['sp1'], data['sp2']]
    zeronn = ZeroNN(verbose=5, epoch=40, path=mkdir('test_train/2'),
                            save_epochs=10, batch_size=512)
    zeronn.fit(data[0][1000:], data[1][1000:], data[2][1000:], 0.2)
    print(zeronn.run_eval(data[0][:1000], data[1][:1000], data[2][:1000]))

    folder_data  = r'F:\Software\vspro\NInRow\NInRow\zero_nns115\godd_mod\103194_1'
    data = [np.load(join(folder_data, npfn('selfplay'+str(i)))) for i in range(3)]

    print(zeronn.run_eval(data[0][:1000], data[1][:1000], data[2][:1000]))


def test_train2():
    path = r'F:\Software\vspro\NInRow\NInRow\test554\selfplay\selfplay54452.npz'
    data = np.load(path)
    print(data['sp0'].shape, data['sp1'].shape, )
    # exit()
    zeronn = ZeroNN(verbose=10, epoch=80, path=mkdir(r'F:\Software\vspro\NInRow\NInRow\test554\NNs'),
                                save_epochs=40, batch_size=256)
    zeronn.fit(data['sp0'], data['sp1'], data['sp2'], 0.1)
    init = np.zeros([1,5,5,4])
    pred = zeronn.predict(init)
    print(pred[0])
    print(pred[1])


def test_symmetry():
    mkdir(r'F:\Software\vspro\NInRow\NInRow\test')
    zeronn = ZeroNN(verbose=10, epoch=80, path=mkdir(r'F:\Software\vspro\NInRow\NInRow\test\test_sym'),
                                    save_epochs=40, batch_size=2)
    num_samples = 4
    rows = 5
    cols = 5
    channel = 4
    x = np.zeros([rows, cols, channel])
    x[0][1][0] = 1
    X = np.repeat([x], num_samples, 0)
    print(X.shape)
    Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
    Y_policy = np.random.rand(num_samples,rows*cols)
    zeronn.fit(X,Y_policy,Y_value)


if __name__=='__main__':
    # main_sim_train()
    # pure_train()
    # test_copy_possible_params()
    # train_comp_kern(int(sys.argv[1]))
    # train_init_net2()
    # test_train2()
    test_symmetry()

