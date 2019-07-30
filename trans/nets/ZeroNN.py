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
from utils.data import *
from global_defs import *
from nets.CNN import CNN
from utils import log
import copy
from scipy import misc
from tensorflow.contrib.slim import nets
from nets import CNN_structures


os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'


class ZeroNN:
    def __init__(self, 
                 common_cnn=CNN_structures.zeronn8, 
                 policy_cnn=CNN.Params([[2,1]], []), 
                 value_cnn=CNN.Params([[1,1],1], [256]), 
                 kp=0.5, lr_init=0.0001, lr_dec_rate=0.95, batch_size=256, ckpt_idx=-1, save_epochs=2,
                 epoch=10, verbose=None, act=tf.nn.relu, l2=1e-4, path=None, lock_model_path=None,
                 num_samples=None, 
                 logger=None):
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
        self.path = None if path is None else mkdir(path)
        self.logger = log.Logger(None if self.path is None else join(self.path, logfn('ZeroNN-' + curr_time_str())), 
                                 verbose is not None)
        if logger is not None:
            self.logger = logger
        self.sess = None
        self.ts = {}
        self.ckpt_idx = ckpt_idx

        # collections of variables
        self.var_names = ['x','kp', 'y_value','y_policy', 'is_train', 
        'loss_policy', 'loss_value', 'pred_value','acc_value', 
        'pred_policy', 'loss_l2', 'loss_total', 'global_step','train_step']
        self.trained_model_paths = []

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
                self.ts[ts] = tf.get_collection(ts)[0]

    def __str__(self):
        return "ZeroNN"
        return "ZeroNN-- common_cnn: {} \tfcs: {} \tkp: {} \tlr_init: {} \tlr_dec_rate: {} \tbatch_size: {} \tepoch: {} \tact: {}".format(
            self.common_cnn, self.fcs, self.kp, self.lr_init, self.lr_dec_rate, self.batch_size, self.epoch, str(self.act).split(' ')[1] if self.act is not None else 'NONE')

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

    def fit(self, X, Y_policy, Y_value, reserve_test=None, refresh_saving=False):
        """
        If you wanna extract test set automatically, set reserve_test the ratio for test set.
        X should be a n*rows*cols*channels, where channels is number of histories.
        """
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
        weights = [0.1,0.2,0.3,0.2,0.1]
        len_w = len(weights)
        tl2br = np.diag(weights)
        filter_tl2br = tf.constant_initializer(tl2br.reshape([len_w,len_w,1,1]))
        filter_tr2bl = tf.constant_initializer(np.rot90(tl2br).reshape([len_w,len_w,1,1]))
        l2r = np.array([0 for _ in range(len_w)] * (len_w//2) + weights + [0 for _ in range(len_w)] * (len_w//2)).reshape([len_w,len_w])
        filter_l2r = tf.constant_initializer(l2r.reshape([len_w,len_w,1,1]))
        filter_t2b = tf.constant_initializer(np.rot90(l2r).reshape([len_w,len_w,1,1]))
        # print(l2r.reshape([len_w,len_w,1,1])[:,:,0,0])
        # print(np.rot90(l2r).reshape([len_w,len_w,1,1])[:,:,0,0])
        # exit(0)
        x = tf.concat([x] + 
                      [slim.conv2d(x[:,:,:,(ch-1):ch], 1, len_w, 
                                   weights_initializer=init,trainable=False)
                       for ch in [1,2] for init in 
                       [filter_tl2br, filter_tr2bl, filter_l2r, filter_t2b]], 
                      axis=3)
        lastdim = int(x.shape[-1])
        # the codes below performs data augmentation
        def augment():
            y_policy_trans = tf.reshape(y_policy, [-1, rows, cols, 1])
            # x and y_policy should be stacked, it is not right to rotate x only!
            x_y_policy = tf.concat([x, y_policy_trans], axis=3)
            x_y_policy = self.tf_random_rotate90(x_y_policy)
            def flip(img):
                return tf.image.random_flip_left_right(tf.image.random_flip_up_down(img))
            x_y_policy = tf.map_fn(flip, x_y_policy)
            x_trans = x_y_policy[:,:,:,:lastdim]
            y_policy_trans = x_y_policy[:,:,:,lastdim]
            y_policy_trans = slim.flatten(y_policy_trans)
            return [x_trans, y_policy_trans]
        def no_augment():
            return [x, y_policy]
        x_trans, y_policy_trans = tf.cond(is_train, augment, no_augment)
        return x_trans, y_policy_trans

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
        x_trans, y_policy_trans = self.preprocess(x, y_policy, is_train, rows, cols)
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
        # pred_policy = tf.nn.softmax(logits_policy, name='pred_policy')

        # regularization_loss = tf.add_n(tf.losses.get_regularization_losses, name='loss_l2') 
        loss_l2 = tf.losses.get_regularization_loss(scope='loss_l2')
        loss_total = tf.add_n([loss_policy, loss_value, loss_l2], name='loss_total')
        global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0.0), trainable=False)

        # learning rate annealing
        lr = tf.train.exponential_decay(
            self.lr_init,
            global_step,
            (n_xs if self.num_samples is None else self.num_samples) / self.batch_size, 
            self.lr_dec_rate,
            staircase=True)
        lr = tf.reduce_max([lr, 5e-6])
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = slim.learning.create_train_op(
            loss_total,  optimizer, global_step=global_step)
        
        # ensure batch normalization is done before forwarding
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            train_step = control_flow_ops.with_dependencies([updates], train_step, name='train_step')
        
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

    def run_eval(self, X, Y_policy, Y_value):
        """
        run evaluation on the given data
        Evaluations are performed in batches and the batch results are synthesized
        """
        if self.sess is None:
            if not self.init_sess(refresh_saving=False):
                raise Exception("Error: trying to evaluate without trained network")
        loss_policy_sum = 0.0
        loss_value_sum = 0.0
        loss_total_sum = 0.0
        acc_value_sum = 0.0
        correct_preds_value = 0.0
        feed_dict = {self.ts['kp']: 1.0, self.ts['is_train']: False}
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
            [loss_policy, 
             loss_value, 
             loss_total, 
             acc_value] = self.sess.run(
                [self.ts['loss_policy'], 
                 self.ts['loss_value'], 
                 self.ts['loss_total'], 
                 self.ts['acc_value']],
                 feed_dict=feed_dict)
            mult = batch_idx_next - batch_idx
            # summarize the evaluation
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
            sess = tf.Session()
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
                self.logger.log("Init new parameters")
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
        tf.summary.FileWriter(self.path,sess.graph)
        for i in range(round(self.epoch * self.X.shape[0] / self.batch_size)+1):
            batch_xs, batch_ys_policy, batch_ys_value = self.next_batch()
            feed_dict = {x_t: batch_xs, kp_t: self.kp, y_value_t: batch_ys_value, y_policy_t: batch_ys_policy, is_train_t: True}
            sess.run(train_step_t, feed_dict=feed_dict)
            # for each epoch, refresh logs and models if necessary
            if i % it_pep == 0:
                it_epoch += 1
                if not (it_epoch % self.save_epochs == 0 or it_epoch % self.verbose == 0):
                    continue
                global_step = sess.run(global_step_t, feed_dict=feed_dict)
                train_eval = self.run_eval(self.X, self.Y_policy, self.Y_value)
                test_eval = [-1.0 for i in range(4)] if self.X_te is None \
                      else self.run_eval(self.X_te, self.Y_policy_te, self.Y_value_te)
                self.train_hists.append([global_step//it_pep] + train_eval)
                self.test_hists.append([global_step//it_pep] + test_eval)
                if self.verbose is not None and it_epoch % self.verbose == 0:
                    self.logger.log('\nglobal_step:',global_step, '  epoch:',global_step//it_pep,
                          '\n   items:        [loss_policy,       loss_value,           loss_total,            acc_value]:',
                          '\n   train_eval: ',train_eval, 
                          '\n   test_eval:',  test_eval)
                if self.path is not None and it_epoch % self.save_epochs == 0:
                    path = self.saver.save(sess, self.path + '/model.ckpt', global_step=global_step_t, write_meta_graph=False)
                    if self.verbose  is not None:
                        self.logger.log('model saved in',path)
                    if self.lock_model_path is not None:
                        self.lock_model_path.acquire()
                    self.trained_model_paths.append(path)
                    if self.lock_model_path is not None:
                        self.lock_model_path.release()
                    self.save_hists()

    def init_hists(self, refresh_saving):
        """
        Initialize training histories:
            If no previous histories are found, create new ones,
            else load the existing histories and append new histories to them
        """
        if refresh_saving:
            self.train_hists = []
            self.test_hists = []
            return None
        path_train = join(self.path, npfn('train'))
        path_test = join(self.path, npfn('test'))
        self.train_hists = [] if not exists(path_train) else np.load(path_train).tolist()
        self.test_hists = [] if not exists(path_test) else np.load(path_test).tolist()

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

    def predict(self, X):
        """
        Make prediction using a trained model: ensure a valid model path is provided.
        """
        if self.sess is None:
            if not self.init_sess(refresh_saving=False):
                raise Exception("Error: trying to predict without trained network")
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



def main_sim_train():
    num_samples = 1024
    rows = 5
    cols = 5
    channel = 4
    X = np.random.rand(num_samples,rows,cols, channel)
    Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
    Y_policy = np.random.rand(num_samples,rows*cols)
    clf = ZeroNN(verbose=2, path='ZeroNN_test', batch_size=512)
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


def new_train():
    top_folder = mkdir('new_train_zeronn')
    zeronn = ZeroNN(verbose=10, epoch=200, path=join(top_folder, 'zeronn8'), 
                    save_epochs=20, batch_size=512)
    # zeronn.fit()
    for folder in ['zero_nns885/selfplay/0', 
                   'zero_nns885/selfplay/1_2_3', 
                   'zero_nns885/selfplay/3', 
                   'zero_nns885/selfplay']:
        X = np.load(join(folder, 'selfplay0.npy'))
        Y_policy = np.load(join(folder, 'selfplay1.npy'))
        Y_value = np.load(join(folder, 'selfplay2.npy'))
        zeronn.fit(X, Y_policy, Y_value, 0.1)
        print(folder, 'trained')
    print('trainning finished!')

        
if __name__=='__main__':
    # main_sim_train()
    new_train()




