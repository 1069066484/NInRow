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


class ZeroNNTrainer:
    """
    ZeroNNTrainer performs ZeroNN's training. 
    Tripal threads are used for training:
        1. optimization
        2. evaluator
        3. self-play
    The details are elaborated in the paper 'Mastering the game of Go without human knowledge'.
    """
    def __init__(self, folder, board_rows=5, board_cols=5, n_in_row=4, 
                 mcts_sims=500, self_play_cnt=5000, reinit=True, batch_size=256):
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.self_play_cnt = self_play_cnt
        self.n_in_row = n_in_row
        self.mcts_sims = mcts_sims
        self.folder = mkdir(folder)
        self.folder_NNs = mkdir(join(folder, 'NNs'))
        self.folder_selfplay = mkdir(join(folder, 'selfplay'))
        self.best_player_path = -1
        self.lock_train_data = threading.Lock()
        self.unchkeck_model_paths = []
        self.lock_model_paths = threading.Lock()
        self.lock_model_best = threading.Lock()
        self.batch_size = batch_size
        self.train_data = None if not reinit else self.init_train_data()
        self.model_avail = False

    def init_train_data(self):
        num_samples = self.batch_size
        rows = self.board_rows
        cols = self.board_cols
        channel = 4
        X = np.random.rand(num_samples,rows,cols, channel)
        Y_value = np.random.randint(0,2,[num_samples,1], dtype=np.int8)
        Y_policy = np.random.rand(num_samples,rows*cols)
        return [X, Y_policy, Y_value]

    def train(self):
        optimization = threading.Thread(target=self.optimization)
        evaluator = threading.Thread(target=self.evaluator)
        self_play = threading.Thread(target=self.self_play)
        optimization.start()
        evaluator.start()
        self_play.start()

    def optimization(self):
        zeroNN = ZeroNN(verbose=3,path=self.folder_NNs, ckpt_idx=-1, epoch=6, batch_size=self.batch_size)
        self.unchkeck_model_paths = zeroNN.trained_model_paths
        print('optimization start!')
        while self.self_play_cnt > 0:
            while self.train_data is None:
                time.sleep(5)
            self.lock_train_data.acquire()
            train_data = [self.train_data[0].copy(), self.train_data[1].copy(), self.train_data[2].copy()]
            self.lock_train_data.release()
            zeroNN.fit(train_data[0],train_data[1],train_data[2])
            self.model_avail = True

    def evaluator(self):
        while not self.model_avail:
            time.sleep(5)
        print('evaluator start!')
        while self.self_play_cnt > 0:
            self.lock_model_paths.acquire()
            if len(self.unchkeck_model_paths) == 0:
                self.lock_model_paths.release()
                time.sleep(5)
                continue
            path_to_check = self.unchkeck_model_paths.pop()
            self.lock_model_paths.release()
            zeroNN_best = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            zeroNN_to_check = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=path_to_check)
            mcts1 = Mcts(0,0,zeroNN=zeroNN_best,max_acts_=self.mcts_sims)
            mcts2 = Mcts(0,0,zeroNN=zeroNN_to_check,max_acts_=self.mcts_sims)
            winrate1, winrate2, tie_rate, ai_hists = \
                eval_mcts(self.board_rows, self.board_cols, self.n_in_row, mcts1, mcts2, False, 200, False)
            if winrate1 - winrate2 > 0.10:
                print('evaluator:',path_to_check, 'defeat' , self.best_player_path, 'by', winrate1 - winrate2)
                self.lock_model_best.acquire()
                self.best_player_path = path_to_check
                self.lock_model_best.release()

    def self_play(self):
        self.self_play_cnt /= 200
        while not self.model_avail:
            time.sleep(5)
        print('self_play start!')
        while self.self_play_cnt > 0:
            zeroNN1 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            zeroNN2 = ZeroNN(verbose=False,path=self.folder_NNs, ckpt_idx=self.best_player_path)
            best_player_path = self.best_player_path
            while True and self.self_play_cnt > 0:
                self.self_play_cnt -= 1
                mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=self.mcts_sims)
                mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=self.mcts_sims)
                winrate1, winrate2, tie_rate, ai_hists = \
                    eval_mcts(self.board_rows, self.board_cols, self.n_in_row, mcts1, mcts2, False, 200, True)
                print('self_play:',winrate1, winrate2 , tie_rate)
                ai_hists = self.hists2enhanced_train_data(ai_hists)
                self.lock_train_data.acquire()
                if len(self.train_data) == self.batch_size:
                    self.train_data = ai_hists
                else:
                    self.train_data = [np.vstack([self.train_data[0], ai_hists[0]]), 
                                       np.vstack([self.train_data[1], ai_hists[1]]),
                                       np.vstack([self.train_data[2], ai_hists[2]])]\
                                           if self.train_data is not None else ai_hists
                if len(self.train_data[0]) > 8000:
                    for i in range(3):
                        self.train_data[i] = self.train_data[i][:4000]
                self.lock_train_data.release()
                self.lock_model_best.acquire()
                find_new_best = self.best_player_path == best_player_path
                self.lock_model_best.release()
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
            for i in range(len(hist[0])):
                X.append(hist[1][i])
                Y_policy.append(hist[0][i])
                Y_value.append([0 if hist[2] is None else int(hist[2] == i % 2)])

                X.append(reversed_eval_board(hist[1][i]))
                Y_policy.append(hist[0][i])
                Y_value.append([0 if hist[2] is None else int(hist[2] != i % 2)])
        return [np.array(X, dtype=np.int8), np.array(Y_policy).reshape(-1,self.board_rows * self.board_cols), np.array(Y_value)]


def main():
    trainer = ZeroNNTrainer(FOLDER_ZERO_NNS)
    trainer.train()


def hists_test():
    pass

if __name__=='__main__':
    main()




