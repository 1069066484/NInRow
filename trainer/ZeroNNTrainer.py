# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: ZeroNN's training pipeline is implemented here.
zero_nns885_3/NNs/model.ckpt-31266
"""
import sys
from global_defs import *
from mcts import MctsPuct
import threading
import time 
from games.game import *
from nets.ZeroNN import *
from utils.data import *
import copy
import random


acquire_lock = threading.Lock
acquire_acc_unit = threading.Thread


MctsPuct.CHECK_DETAILS = False
rcn = '885' + (('_' + sys.argv[1]) if len(sys.argv) > 1 else '')


class ZeroNNTrainer:
    """
    ZeroNNTrainer performs ZeroNN's training. 
    Tripal groups of threads are used for training:
        1. optimization.
        2. evaluator. Evaluation must be fast; the best situation is that once a new model is trained, the it can be evaluated.
        3. self-play. Self-play should play games using the best generated model.
    And we use multiple threads for evaluator and self-play.
    The details are elaborated in the paper 'Mastering the game of Go without human knowledge'.
    """
    def __init__(self, folder, board_rows=int(rcn[0]), board_cols=int(rcn[1]), n_in_row=int(rcn[2]), 
                 train_ratio=0.15, train_size=128*1024, mcts_sims=512, self_play_cnt=10000, reinit=False, batch_size=512, verbose=True, n_eval_threads=1, best_player_path=-1, n_play_threads=1, plays=10, start_nozero=False, do_opt=True):
        self.board_rows = board_rows if board_rows > n_in_row else board_rows + 10
        self.board_cols = board_cols if board_cols > n_in_row else board_cols + 10
        self.self_play_cnt = self_play_cnt
        self.n_play_threads = n_play_threads
        self.n_in_row = n_in_row
        self.mcts_sims = mcts_sims
        self.folder = mkdir(folder)
        self.train_size = train_size
        self.plays = plays
        self.n_eval_threads = n_eval_threads
        self.train_ratio = train_ratio
        self.folder_NNs = mkdir(join(folder, 'NNs'))
        self.folder_selfplay = mkdir(join(folder, 'selfplay'))
        self.best_player_path = best_player_path
        self.lock_train_data = acquire_lock()
        self.unchecked_model_paths = []
        self.lock_model_paths = acquire_lock()
        self.lock_model_best = acquire_lock()
        self.batch_size = batch_size
        self.train_data = None if not reinit else self.init_train_data()
        self.model_avail = not reinit and not start_nozero
        self.data_avail = False
        self.resign_val = 0.99
        self.do_opt = do_opt
        self.hand_val = 1.0
        self.curr_generation = 0
        self.data_path = [join(self.folder_selfplay, npfn('selfplay' + str(i))) for i in range(3)]
        self.loss_hists = []
        self.path_loss_hists = join(self.folder_selfplay, npfn('selfplay' + '_loss'))
        self.logger = log.Logger(
            join(self.folder, logfn('ZeroNNTrainer-' + curr_time_str())), verbose)
        self.nozero_mcts_sims = int(mcts_sims * 2)
        # self.nozero_mcts is the initial MCTS
        # self.nozero_mcts use no ZeroNN for seaching and we use it to generate initial training data
        # instead of using zeroNNs with randomized parameters
        # After the first generation of zeroNN trained, self.nozero_mcts would be set None since it is not needed anymore
        if reinit:
            self.nozero_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=0.2,noise=0.1, hand_val=self.hand_val)
        else:
            if exists(self.data_path[0]) and exists(self.data_path[1]) and exists(self.data_path[1]):
                self.train_data = [np.load(p) for p in self.data_path]
            if exists(self.path_loss_hists):
                self.loss_hists = np.load(self.path_loss_hists).tolist()
                self.curr_generation = self.loss_hists[-1][0]
            else:
                self.loss_hists = []
            self.nozero_mcts = None
        if start_nozero:
            self.nozero_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=0.2,noise=0.1, hand_val=self.hand_val)

    def init_train_data(self):
        """
        Just ignore the function. It help build a randomized model.
        """
        num_samples = 1
        rows = self.board_rows
        cols = self.board_cols
        channel = 4
        X = np.random.rand(num_samples,rows,cols, channel).astype(np.bool)
        Y_value = np.random.randint(0,2,[num_samples,1])
        Y_policy = np.random.rand(num_samples,rows*cols)
        return [X, Y_policy, Y_value]

    def train(self):
        """
        Entrance codes
        """
        threads = ([acquire_acc_unit(target=self.optimization)] if self.do_opt else []) \
            + [acquire_acc_unit(target=self.console)] + \
            [acquire_acc_unit(target=self.self_play) for _ in range(self.n_play_threads)] + \
            [acquire_acc_unit(target=self.evaluator) for _ in range(self.n_eval_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.logger.log('trained')

    def console(self):
        """
        Use console to help training
        """
        while True:
            cmd = input()
            try:
                if cmd == 'md':
                    folder = input('Input the folder name where the three data file(np file) exists. '+
                     'The files should be named [selfplay0.npy], [selfplay1.npy] and [selfplay2.npy]:\n')
                    try:
                        data_path = [join(folder, npfn('selfplay' + str(i))) for i in range(3)]
                        train_data = [np.load(p) for p in data_path]
                        with self.lock_train_data:
                            self.train_data = [np.vstack([self.train_data[0], train_data[0]]).astype(np.bool), 
                                np.vstack([self.train_data[1], train_data[1]]),
                                np.vstack([self.train_data[2], train_data[2]])]
                        self.data_avail = True
                        print("load new data succeessfully, data size = ", len(self.train_data[0]))
                    except:
                        print("md error: folder or files do not exist or are invalid")
                elif cmd == 'pl':
                    plays = input("Input the number of games for one self-play, it should be a positive even number:\n")
                    def plerr():
                        print("pl error: ",plays," is invalid to be a self-play number")
                    try:
                        plays = int(plays)
                        if plays > 0 and plays % 2 == 0:
                            self.plays = plays
                            print("self.plays -> ", self.plays)
                        else:
                            plerr()
                    except:
                        plerr()
                elif cmd == 'bs':
                    batch_size = input("Input the number of batch size, it should be a positive number:\n")
                    def bserr():
                        print("bs error")
                    try:
                        batch_size = int(batch_size)
                        if batch_size > 0:
                            self.batch_size = batch_size
                            print("Batch size reset to" ,batch_size)
                        else:
                            bserr()
                    except:
                        bserr()
                elif cmd == 'bp':
                    best_player_path = input("Input num best player path, th current is " + str(self.best_player_path) + ":\n")
                    if os.path.exists(best_player_path + '.index'):
                        self.best_player_path = best_player_path
                        print("best player path -> ", self.best_player_path)
                    else:
                        print("bp error,",best_player_path,'is invalid')
                elif cmd == 'nm':
                    def nmerr():
                        print("nm error")
                    nozero_mcts_sims = input("Input number of simulations of nozero_mcts's search. A non-positive number means deactivation\n" + " Currently, nozero_mcts is " + str(self.nozero_mcts) + ". And nozero_mcts_sims is " + str(self.nozero_mcts_sims) + ":\n")
                    try:
                        nozero_mcts_sims = int(nozero_mcts_sims)
                        if nozero_mcts_sims > 0:
                            self.nozero_mcts_sims = nozero_mcts_sims
                            self.nozero_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=0.2,noise=0.1, hand_val=self.hand_val)
                            print("nozero_mcts activated. And nozero_mcts_sims is ", self.nozero_mcts_sims)
                        else:
                            self.nozero_mcts = None
                            print("nozero_mcts deactivated. And nozero_mcts_sims is ", self.nozero_mcts_sims)
                    except:
                        nmerr()
                elif cmd == 'op':
                    print("Start an optimization now")
                    self.data_avail = True
                elif cmd == 'pc':
                    self_play_cnt = input("Input number of play count:\n")
                    try:
                        self_play_cnt = int(self_play_cnt)
                        self.self_play_cnt = max(self_play_cnt,0)
                        print("self.self_play_cnt -> ", self.self_play_cnt)
                    except:
                        print("pc error")
                elif cmd == 'te':
                    # use te will clear the screen
                    game = Game(self.board_rows,self.board_cols,self.n_in_row, Game.Player.AI, Game.Player.human, collect_ai_hists=False)
                    zeroNN1 = ZeroNN(path=join(self.folder_NNs), ckpt_idx=-1)
                    game.players[0].mcts.zeroNN = zeroNN1
                    game.players[0].mcts.max_acts = self.mcts_sims
                    game.start(graphics=True)
                elif cmd == 'di':
                    def dierr():
                        print("di error")
                    discard = input("Input number of data to discard, the current number is " + str(len(self.train_data[0])) +":\n")
                    try:
                        discard = int(discard)
                        if discard <= 0 or discard > len(self.train_data[0]) - 1:
                            dierr()
                        for i in range(3):
                            self.train_data[i] = self.train_data[i][discard:]
                        print("Discard successfully! The current number is ", len(self.train_data[0]))
                    except:
                        dierr()
                elif cmd == 'hv':
                    def hverr():
                        print('hv errpr')
                    hand_val = input("Input hand_val, the current number is " + str(self.hand_val) +":\n")
                    try:
                        self.hand_val = float(hand_val)
                    except:
                        hverr()
                elif cmd == 'sv':
                    for i in range(3):
                        np.save(self.data_path[i], self.train_data[i])
                elif cmd == 'pt':
                    print('Folder=',self.folder)
                else:
                    print("command error. (cmd=",cmd ,")")
            except:
                print("Unknown console error!")

    def optimization(self):
        with self.lock_model_paths:
            zeroNN = ZeroNN(verbose=2,path=self.folder_NNs, ckpt_idx=-1, num_samples=self.train_size,
                           epoch=3, batch_size=self.batch_size, save_epochs=4, logger=self.logger)
            self.unchecked_model_paths = zeroNN.trained_model_paths
        self.logger.log('optimization start!')
        while self.self_play_cnt > 0:
            while self.train_data is None or\
               not self.data_avail or\
               self.train_ratio * len(self.train_data[0]) * 0.8 < self.batch_size:
                time.sleep(10)
            # Wait for the models to be evaluated
            # Better models need to be selected to generate better data
            # remove old models to ease burden of evaluator
            while len(self.unchecked_model_paths) > 10 and np.random.rand() < 0.99:
                time.sleep(20)
                with self.lock_model_paths:
                    self.unchecked_model_paths.remove(
                        self.unchecked_model_paths[round(np.random.rand() * 8)])
            # given time slices for the other two threads
            with self.lock_train_data:
                train_data = [self.train_data[0].copy(), 
                              self.train_data[1].copy(), 
                              self.train_data[2].copy()]
            # save the training data in case that we need to use them to continue training
            for i in range(3):
                np.save(self.data_path[i], self.train_data[i])
            print(len(self.train_data[0]), 'saved')
            # select some playing histories to train to control overfitting
            nonrep_rand_nums = non_repeated_random_nums(len(train_data[0]), round(self.train_ratio * len(train_data[0])))
            zeroNN.fit(train_data[0][nonrep_rand_nums],
                       train_data[1][nonrep_rand_nums],
                       train_data[2][nonrep_rand_nums], 0.1)
            self.data_avail = False
            zeroNN.epoch = zeroNN.verbose = zeroNN.save_epochs = 10
            self.model_avail = True
            while not self.data_avail:
                time.sleep(30)

    def evaluator(self):
        while not self.model_avail:
            time.sleep(5)
        self.lock_model_paths.acquire()
        while len(self.unchecked_model_paths) != 0:
            self.unchecked_model_paths.pop()
        self.lock_model_paths.release()
        # try to test checkpoints as different as possible
        time.sleep(round(np.random.rand()*30*self.n_eval_threads+1))
        self.logger.log('evaluator start!')
        while self.self_play_cnt > 0 or len(self.unchecked_model_paths) > 5:
            # use 'with' to lock to avoid forgetting to release it 
            with self.lock_model_paths:
                # wait for latest trained model
                if len(self.unchecked_model_paths) < 2:
                    time.sleep(30)
                    continue
                path_to_check = self.unchecked_model_paths.pop()
                if len(self.unchecked_model_paths) > 5:
                    if np.random.rand() < 0.5:
                        self.unchecked_model_paths.pop()
                    else:
                        path_to_check = self.unchecked_model_paths.pop()
            self.logger.log('evaluator:',self.best_player_path, 'VS' , path_to_check, '...')
            if self.nozero_mcts is None:
                best_mcts = Mcts(
                    0,0,zeroNN=ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path),
                    max_acts_=self.mcts_sims,const_temp=0,noise=0.1, resign_val=self.resign_val, hand_val=self.hand_val)
            else:
                # When self.nozero_mcts is not None, the first generation of zeroNN is not generated yet
                # We double number of simulations since MCTS without zeroNN can make a faster searching,
                # which also means any trained model is able to defeat MCTS without zeroNN using doule simulations
                best_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=0.2,noise=0.1, hand_val=self.hand_val)
            zeroNN_to_check = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=path_to_check)
            mcts2 = Mcts(0,0,zeroNN=zeroNN_to_check,max_acts_=self.mcts_sims,const_temp=0,noise=0.1, resign_val=self.resign_val, hand_val=self.hand_val)
            
            # the evaluation must be fast to select the best model
            # play only several games, but require the player to check have a overwhelming advantage over the existing player
            # if the best player is defeated, then the player to check can take the first place
            winrate1, winrate2, tie_rate, _ = \
                eval_mcts(self.board_rows, self.board_cols, self.n_in_row, best_mcts, mcts2, False, [3,1], False)
            self.logger.log('evaluator:',self.best_player_path, 'VS' , path_to_check,'--', winrate1,'-',winrate2,'-',tie_rate)
            time.sleep(5)
            # if the new player wins 3 out of 4 and draws in one game, replace the best player with it
            if winrate2 > 0.4 and winrate1 < 0.01:
                self.curr_generation += 1
                self.logger.log('evaluator:',path_to_check, 'defeat' , self.best_player_path, 'by', winrate2 - winrate1)
                self.logger.log(path_to_check, 'becomes generation' , self.curr_generation)
                with self.lock_model_best:
                    self.best_player_path = path_to_check
                self.nozero_mcts = None

    def self_play(self):
        while not self.model_avail and self.nozero_mcts is None:
            time.sleep(5)
        time.sleep(round(np.random.rand()*60*self.n_play_threads+1))
        self.logger.log('self_play start!')
        plays = self.plays
        while self.self_play_cnt > 0:
            zeroNN1 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            zeroNN2 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            best_player_path = self.best_player_path
            # we do not lock for self_play_cnt
            while self.self_play_cnt > 0:
                self.self_play_cnt -= plays
                # decay resign_val
                # rookies should always play the game to the end while masters are allowed to resign at an earlier stage
                self.resign_val = max(0.75, self.resign_val - self.resign_val * 0.00002 * plays)
                self.logger.log('self_play:','self_play_cnt=',self.self_play_cnt,' self.resign_val=',self.resign_val)
                # Create two identical players to 'self play'
                if self.nozero_mcts is not None:
                    mcts1 = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=1,noise=0.2, 
                                 resign_val=0.99, temp2zero_moves=3, hand_val=self.hand_val)
                    mcts2 = Mcts(0,0,zeroNN=None,max_acts_=self.nozero_mcts_sims,const_temp=1,noise=0.2, 
                                 resign_val=0.99, temp2zero_moves=3, hand_val=self.hand_val)
                else:
                    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=self.mcts_sims, const_temp=1, 
                                 temp2zero_moves=3, noise=0.2, resign_val=self.resign_val, hand_val=self.hand_val)
                    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=self.mcts_sims, const_temp=1, 
                                 temp2zero_moves=3, noise=0.2, resign_val=self.resign_val, hand_val=self.hand_val)
                t = time.time()
                winrate1, winrate2, tie_rate, ai_hists = \
                    eval_mcts(self.board_rows, self.board_cols, self.n_in_row, mcts1, mcts2, False, plays//2, True)
                ai_hists = self.hists2enhanced_train_data(ai_hists)
                # Evaluate how the zeroNN works on the latest played game.
                # This is the real test data since the data are not feeded for zeroNN's training yet so we need to save the 
                # evaluations.
                if self.nozero_mcts is None:
                    eval = zeroNN1.run_eval(ai_hists[0], ai_hists[1], ai_hists[2])
                    self.logger.log(
                            'self-play in ' + str(time.time() - t) + 's\n',
                            'sp items:        [loss_policy,       loss_value,           loss_total,            acc_value]:',
                            '\n   eval:         ',eval)
                    self.loss_hists.append([self.curr_generation] + eval)
                # Append the latest data to the old.
                with self.lock_train_data:
                    if self.train_data is None or len(self.train_data) == self.batch_size:
                        self.train_data = ai_hists
                    else:
                        self.train_data = [np.vstack([self.train_data[0], ai_hists[0]]).astype(np.bool), 
                                           np.vstack([self.train_data[1], ai_hists[1]]),
                                           np.vstack([self.train_data[2], ai_hists[2]])]\
                                               if self.train_data is not None else ai_hists
                    # save the training data in case that we need to use them to continue training
                    for i in range(3):
                        np.save(self.data_path[i], self.train_data[i])
                    # Discard some old data since our memory is running out
                    if len(self.train_data[0]) > self.train_size + 1:
                        for i in range(3):
                            self.train_data[i] = self.train_data[i][-round(self.train_size * 0.6+1):]
                self.logger.log('self_play:',winrate1, winrate2 , tie_rate,'  new data size=', 
                                ai_hists[0].shape, '   total data:', self.train_data[0].shape)
                self.data_avail = True
                with self.lock_model_best:
                    find_new_best = (self.best_player_path != best_player_path)
                if find_new_best and len(self.train_data[0]) > 10000:
                    # Discard some old data since a new best player is trained, we need to use data of games played by
                    # it to train new models
                    with self.lock_train_data:
                        len_train_data0 = len(self.train_data[0])
                        for i in range(3):
                            self.train_data[i] = self.train_data[i][-round(len_train_data0 * 0.6+1):]
                    break


    def try_enh(self, X, Y_policy, Y_value):
        """
        it takes about 0.005s to dispose one data item
        """
        if len(X) == 0:
            return
        x = X[-1]
        y_policy = Y_policy[-1]
        y_value = Y_value[-1]
        def enh(dir):
            x_enh = x.copy()
            for i in range(3):
                # print("\nbefore\n",x[:,:,i])
                x_enh[:,:,i] = matMove(x[:,:,i], dir)
                # print(x_enh[:,:,i])
            y_policy_enh = matMove(y_policy, dir)
            y_policy_enh *= 1.0 / np.sum(y_policy_enh)
            # print(np.sum(y_policy_enh))
            X.append(x_enh)
            Y_policy.append(y_policy_enh)
            Y_value.append(y_value)
        s = np.sum(x[:,:,:3])
        if s == 0:
            return
        for step in [1,2,3,4]:
            for sx, sx2, dir in \
                            [[np.sum(x[:(2+step),:,:3]), np.sum(x[-2:,:,:3]), [-step,0]],
                            [np.sum(x[:,:(2+step),:3]),  np.sum(x[:,-2:,:3]), [0,-step]],
                            [np.sum(x[-(2+step):,:,:3]), np.sum(x[:2,:,:3]), [step,0]],
                            [np.sum(x[:,-(2+step):,:3]), np.sum(x[:,:2,:3]), [0,step]]]:
                if sx < 1e-6 and (s <= 5 or  sx2 < 1e-6) and\
                   (random.random() * random.random() * self.curr_generation < 1):
                    enh(dir)


    def hists2enhanced_train_data(self, ai_hists):
        """
        Convert game histories to training data
        Training data are saved with a very low cost of space
        """
        X = []
        Y_policy = []
        Y_value = []
        for hist in ai_hists:
            rate = 0.01
            for i in range(len(hist[0])):
                X.append(hist[1][i])
                Y_policy.append(hist[0][i])
                # the begining two steps should be treated as tie
                Y_value.append([0 if (hist[2] is None or i < 2) else 
                                (int(hist[2] != i % 2) * 2 - 1) * rate])
                self.try_enh(X, Y_policy, Y_value)
                rate = min(rate + 0.3 * (i % 2), 1.0)
        return [np.array(X, dtype=np.bool), 
                np.array(Y_policy, dtype=np.float32).reshape(-1,self.board_rows * self.board_cols), 
                np.array(Y_value)]


def main():
    trainer = ZeroNNTrainer(FOLDER_ZERO_NNS+rcn)
    trainer.train()


def eval_test():
    zeroNN1 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    zeroNN2 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=100)
    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=100)
    winrate1, winrate2, tie_rate, ai_hists = \
        eval_mcts(5, 5, 4, mcts1, mcts2, True, 1, True)


def try_enh(X, Y_policy, Y_value):
    if len(X) == 0:
        return
    x = X[-1]
    y_policy = Y_policy[-1]
    y_value = Y_value[-1]
    def enh(dir):
        x_enh = x.copy()
        for i in range(3):
            # print("\nbefore\n",x[:,:,i])
            x_enh[:,:,i] = matMove(x[:,:,i], dir)
            # print(x_enh[:,:,i])
        y_policy_enh = matMove(y_policy, dir)
        y_policy_enh *= 1.0 / np.sum(y_policy_enh)
        # print(np.sum(y_policy_enh))
        X.append(x_enh)
        Y_policy.append(y_policy_enh)
        Y_value.append(y_value)
    s = np.sum(x[:,:,0])
    for step in [1,2,3]:
        for sx, sx2, dir in \
                        [[np.sum(x[:(2+step),:,:3]), np.sum(x[-2:,:,:3]), [-step,0]],
                        [np.sum(x[:,:(2+step),:3]),  np.sum(x[:,-2:,:3]), [0,-step]],
                        [np.sum(x[-(2+step):,:,:3]), np.sum(x[:2,:,:3]), [step,0]],
                        [np.sum(x[:,-(2+step):,:3]), np.sum(x[:,:2,:3]), [0,step]]]:
            if sx < 1e-6 and (s <= 2 or  sx2 < 1e-6):
                enh(dir)


def test_try_enh():
    X = np.zeros([4,6,4])
    X[:,:,0] = np.array(
        [[0,0,1,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,1] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,2] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X[:,:,3] = np.array(
        [[0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0]]
        )
    X = [X]
    Y_policy = [
        np.array(
            [[0,    0,      0,      0,      0.1,    0],
            [0.1,   0,      0,      0,      0,      0.1],
            [0,     0,      0.3,      0,      0,      0],
            [0,     0.2,    0.2,    0,      0,      0]]
            )
        ]
    Y_value = [-0.5]
    t = time.time()
    try_enh(X, Y_policy, Y_value)
    for i in range(100):
        X.pop()
        Y_policy.pop()
        Y_value.pop()
        try_enh(X, Y_policy, Y_value)
    print('time=',time.time() - t)
    print(len(X), X[0].shape)
    for i in range(4):
        print(X[-1][:,:,i])
    print('\n')
    print(Y_policy[-1])
    print(Y_value[-1])


if __name__=='__main__':
    # test_try_enh()
    main()




