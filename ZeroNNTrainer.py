# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: ZeroNN's training logits are implemented here.
"""
from global_defs import *
import threading
import time
from game_utils import *
from ZeroNN import *
from data_utils import *
import copy


class ZeroNNTrainer:
    """
    ZeroNNTrainer performs ZeroNN's training. 
    Tripal threads are used for training:
        1. optimization.
        2. evaluator. Evaluation must be fast; the best situation is that once a new model is trained, the it can be evaluated.
        3. self-play. Self-play should better play using the best generated model.
    And we use multiple threads for evaluator and self-play.
    The details are elaborated in the paper 'Mastering the game of Go without human knowledge'.
    """
    def __init__(self, folder, board_rows=6, board_cols=6, n_in_row=4, train_ratio=0.9, train_size=128*16,
                 mcts_sims=128, self_play_cnt=4000, reinit=True, batch_size=128, verbose=True, n_eval_threads=2,
                n_play_threads=3):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.self_play_cnt = self_play_cnt
        self.n_play_threads = n_play_threads
        self.n_in_row = n_in_row
        self.mcts_sims = mcts_sims
        self.folder = mkdir(folder)
        self.train_size = train_size
        self.n_eval_threads = n_eval_threads
        self.train_ratio = train_ratio
        self.folder_NNs = mkdir(join(folder, 'NNs'))
        self.folder_selfplay = mkdir(join(folder, 'selfplay'))
        self.best_player_path = -1
        self.lock_train_data = threading.Lock()
        self.unchkeck_model_paths = []
        self.lock_model_paths = threading.Lock()
        self.lock_model_best = threading.Lock()
        self.batch_size = batch_size
        self.train_data = None if not reinit else self.init_train_data()
        self.model_avail = not reinit
        self.data_avail = False
        self.resign_val = 0.99
        self.logger = log.Logger(
            join(self.folder, logfn('ZeroNNTrainer-' + curr_time_str())), verbose)
        if reinit:
            self.best_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.mcts_sims*2,const_temp=0.2,noise=0.1)
        else:
            self.best_mcts = None

    def init_train_data(self):
        num_samples = self.batch_size + 1
        rows = self.board_rows
        cols = self.board_cols
        channel = 4
        X = np.random.rand(num_samples,rows,cols, channel)
        Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
        Y_policy = np.random.rand(num_samples,rows*cols)
        return [X, Y_policy, Y_value]

    def train(self):
        threads = [threading.Thread(target=self.optimization)] + \
            [threading.Thread(target=self.self_play) for _ in range(self.n_play_threads)] + \
            [threading.Thread(target=self.evaluator) for _ in range(self.n_eval_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.logger.log('trained')

    def optimization(self):
        with self.lock_model_paths:
            zeroNN = ZeroNN(verbose=2,path=self.folder_NNs, ckpt_idx=-1, num_samples=self.train_size,
                           epoch=3, batch_size=self.batch_size, save_epochs=4, logger=self.logger)
            self.unchkeck_model_paths = zeroNN.trained_model_paths
        self.logger.log('optimization start!')
        while self.self_play_cnt > 0:
            while self.train_data is None or not self.data_avail or len(self.train_data[0]) < self.batch_size:
                time.sleep(10)
            # Wait for the models to be evaluated
            # Better models need to be selected to generate better data
            # remove old models to ease burden of evaluator
            while len(self.unchkeck_model_paths) > 10 and np.random.rand() < 0.99:
                time.sleep(20)
                with self.lock_model_paths:
                    self.unchkeck_model_paths.remove(self.unchkeck_model_paths[round(np.random.rand() * 8)])
            # given time slices for the other two threads
            self.lock_train_data.acquire()
            train_data = [self.train_data[0].copy(), self.train_data[1].copy(), self.train_data[2].copy()]
            self.lock_train_data.release()
            zeroNN.fit(train_data[0],train_data[1],train_data[2], 0.1)
            zeroNN.epoch = 11
            zeroNN.verbose = 10
            zeroNN.save_epochs = 10
            self.model_avail = True
            while not self.data_avail:
                time.sleep(5)

    def evaluator(self):
        while not self.model_avail:
            time.sleep(5)
        self.lock_model_paths.acquire()
        while len(self.unchkeck_model_paths) != 0:
            self.unchkeck_model_paths.pop()
        self.lock_model_paths.release()
        # try to test checkpoints as different as possible
        time.sleep(round(np.random.rand()*30*self.n_eval_threads+1))
        self.logger.log('evaluator start!')
        while self.self_play_cnt > 0 or len(self.unchkeck_model_paths) > 5:
            self.lock_model_paths.acquire()
            if len(self.unchkeck_model_paths) < 2:
                self.lock_model_paths.release()
                time.sleep(10)
                continue
            path_to_check = self.unchkeck_model_paths.pop()
            if len(self.unchkeck_model_paths) > 5:
                if np.random.rand() < 0.5:
                    self.unchkeck_model_paths.pop()
                else:
                    path_to_check = self.unchkeck_model_paths.pop()
            self.lock_model_paths.release()
            self.logger.log('evaluator:',self.best_player_path, 'VS' , path_to_check)
            if self.best_mcts is None:
                best_mcts = Mcts(
                    0,0,zeroNN=ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path),
                    max_acts_=self.mcts_sims//2,const_temp=0,noise=0.1, resign_val=self.resign_val)
            else:
                best_mcts = Mcts(0,0,zeroNN=None,max_acts_=self.mcts_sims*2//2,const_temp=0.2,noise=0.1)
            zeroNN_to_check = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=path_to_check)
            mcts2 = Mcts(0,0,zeroNN=zeroNN_to_check,max_acts_=self.mcts_sims//2,const_temp=0,noise=0.1, resign_val=self.resign_val)
            # the evaluation must be fast to select the best model
            # play only two games, but giving the first move to the best player
            # if the best player is defeated, then the player to check can take the first player
            winrate1, winrate2, tie_rate, ai_hists = \
                eval_mcts(self.board_rows, self.board_cols, self.n_in_row, best_mcts, mcts2, False, [2,0], False)
            self.logger.log('evaluator:',self.best_player_path, 'VS' , path_to_check,'--', winrate1,'-',winrate2,'-',tie_rate)
            # if the new player wins all, replace the best player with it
            if winrate2 - winrate1 > 0.99:
                self.logger.log('evaluator:',path_to_check, 'defeat' , self.best_player_path, 'by', winrate2 - winrate1)
                with self.lock_model_best:
                    self.best_player_path = path_to_check
                self.best_mcts = None

    def self_play(self):
        while not self.model_avail and self.best_mcts is None:
            time.sleep(5)
        time.sleep(round(np.random.rand()*60*self.n_play_threads+1))
        self.logger.log('self_play start!')
        plays = 4
        while self.self_play_cnt > 0:
            zeroNN1 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            zeroNN2 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            best_player_path = self.best_player_path
            # we do not lock for self_play_cnt
            while self.self_play_cnt > 0:
                self.self_play_cnt -= plays
                
                # decay resign_val
                # when rookies should always play the game to the end
                self.resign_val = max(0.55, self.resign_val - self.resign_val * 0.0005 * plays)
                self.logger.log('self_play:','self_play_cnt=',self.self_play_cnt,' self.resign_val=',self.resign_val)
                if self.best_mcts is not None:
                    mcts1 = Mcts(0,0,zeroNN=None,max_acts_=self.mcts_sims*2,const_temp=1,noise=0.2, resign_val=0.99)
                    mcts2 = Mcts(0,0,zeroNN=None,max_acts_=self.mcts_sims*2,const_temp=1,noise=0.2, resign_val=0.99)
                else:
                    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=self.mcts_sims, const_temp=1, 
                                 temp2zero_moves=3, noise=0.2, resign_val=self.resign_val)
                    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=self.mcts_sims, const_temp=1, 
                                 temp2zero_moves=3, noise=0.2, resign_val=self.resign_val)
                winrate1, winrate2, tie_rate, ai_hists = \
                    eval_mcts(self.board_rows, self.board_cols, self.n_in_row, mcts1, mcts2, False, plays//2, True)
                ai_hists = self.hists2enhanced_train_data(ai_hists)
                self.logger.log('self_play:',winrate1, winrate2 , tie_rate,'  data size=', ai_hists[0].shape)
                self.lock_train_data.acquire()
                if self.train_data is None or len(self.train_data) == self.batch_size:
                    self.train_data = ai_hists
                else:
                    self.train_data = [np.vstack([self.train_data[0], ai_hists[0]]), 
                                       np.vstack([self.train_data[1], ai_hists[1]]),
                                       np.vstack([self.train_data[2], ai_hists[2]])]\
                                           if self.train_data is not None else ai_hists
                if len(self.train_data[0]) > self.train_size + 1:
                    for i in range(3):
                        self.train_data[i] = self.train_data[i][-round(self.train_size * 0.6+1):]
                self.lock_train_data.release()
                self.data_avail = True
                with self.lock_model_best:
                    find_new_best = (self.best_player_path == best_player_path)
                if find_new_best:
                    break

    def reversed_eval_board(self, board):
        board_new = board.copy()
        # the oppenent is the next to move
        board_new[:,:,3] = 1 - board[:,:,3]
        # swap the player to move
        board_new[:,:,0] = board[:,:,1]
        board_new[:,:,1] = board[:,:,0]
        return board_new

    def hists2enhanced_train_data(self, ai_hists):
        X = []
        Y_policy = []
        Y_value = []
        for hist in ai_hists:
            rate = 0.1
            for i in range(len(hist[0])):
                X.append(hist[1][i])
                Y_policy.append(hist[0][i])
                # the begining two steps should be treated ties
                Y_value.append([0 if (hist[2] is None) else 
                                (int(hist[2] != i % 2) * 2 - 1) * rate])
                rate = min(rate + 0.2, 1.0)
        nonrep_rand_nums = non_repeated_random_nums(len(X), round(self.train_ratio * len(X)))
        return [np.array(X, dtype=np.int8)[nonrep_rand_nums], 
                np.array(Y_policy, dtype=np.float)[nonrep_rand_nums].reshape(-1,self.board_rows * self.board_cols), 
                np.array(Y_value)[nonrep_rand_nums]]


def main():
    trainer = ZeroNNTrainer(FOLDER_ZERO_NNS+'664')
    trainer.train()


def eval_test():
    zeroNN1 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    zeroNN2 = ZeroNN(verbose=False,path=mkdir(join(FOLDER_ZERO_NNS, 'NNs')), ckpt_idx=-1)
    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=100)
    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=100)
    winrate1, winrate2, tie_rate, ai_hists = \
        eval_mcts(5, 5, 4, mcts1, mcts2, True, 1, True)


if __name__=='__main__':
    main()




