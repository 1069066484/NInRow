# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: console interface and game board logit.
"""

from math import log, sqrt
import numpy as np
from enum import IntEnum
from mcts.MctsPuct import MctsPuct as Mcts, Grid
# from MctsUct import MctsUct as Mcts, Grid
from games.Termination import *
import os
import sys
import copy


class AI:
    def __init__(self, board, n_in_row):
        self.board = board.board
        self.mcts = Mcts(self.board.shape[0], self.board.shape[1], n_in_row)

    def set_id(self, grid_id):
        self.grid_id = grid_id
        trans_dict = {Board.GridState.empty:Grid.GRID_EMP}
        if grid_id == Board.GridState.p1:
            trans_dict.update({Board.GridState.p1: Grid.GRID_SEL, Board.GridState.p2: Grid.GRID_ENY})
        else:
            trans_dict.update({Board.GridState.p2: Grid.GRID_SEL, Board.GridState.p1: Grid.GRID_ENY})
        self.trans_dict = trans_dict

    def get_valid_action(self):
        board = self.board.copy()
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                board[r][c] = self.trans_dict[board[r][c]]
        return self.mcts.select_action(board)



class Board:
    class GridState(IntEnum):
        empty = 0
        p1 = 1
        p2 = -1
        invalid = 0xff

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros([rows, cols], np.int8)

    def get_loc_state(self, loc):
        #print(self.board)
        if 0 <= loc[0] < self.rows and 0 <= loc[1] < self.cols:
            return self.board[loc[0]][loc[1]]
        return GridState.invalid


class Human:
    def __init__(self, board, n_in_row):
        self.board = board

    def set_id(self, grid_id):
        pass

    def get_valid_action(self):
        while True:
            try:
                s = input("Your move: ").split(',')
                location = [int(s[0]), int(s[1])]
                if self.board.get_loc_state(location) != Board.GridState.empty:
                    raise Exception()
                return location
            except:
                print("invalid move")


class Game:
    class Player(IntEnum):
        AI = 0
        human = 1

    def __init__(self, rows, cols, n_in_row, p1=Player.human, p2=Player.AI, collect_ai_hists=False, use_hists=None, hist_it=0):
        """
        You can use either type Player for initialization or an initialized object
        @collect_ai_hists:  whether or not to collect ai's histories, only available if both players are AI.
        @use_hists: it should be a path that contains n-r-c-4 array or an array like this.
        """
        self.board = Board(rows, cols)
        init_player = [AI, Human]
        self.ps = [p1, p2]
        self.players = [init_player[p](self.board, n_in_row)
                        for p in [p1, p2]]
        self.players[0].set_id(Board.GridState.p1)
        self.players[1].set_id(Board.GridState.p2)
        self.n_in_row = n_in_row
        assert(self.n_in_row <= rows and self.n_in_row <= cols )
        self.collect_ai_hists = collect_ai_hists
        self.all_ai = (p1 + p2 == 0)
        self.use_hists = np.load(use_hists) if isinstance(use_hists, str) else use_hists
        self.hist_it = hist_it
        self.init_replay()
        if collect_ai_hists:
            assert(self.all_ai)
            self.hists_prob = []
            self.hists_board = []

    def init_replay(self):
        if self.use_hists is None:
            return
        try:
            while True:
                if np.sum(self.use_hists[self.hist_it,:,:,:3]) == 0:
                    # print(1)
                    # input("Going to replay at " + str(self.hist_it), '  sum=', np.sum(self.use_hists[self.hist_it,:,:,:3]))
                    self.hist_it += 1
                    return
                self.hist_it += 1
        except:
            raise Exception("Error invalid history, cannot replay")

    def graphics(self, to_print=''):
        """
        Show user interface.
        """
        if sys.platform == 'win32':
            os.system('cls')
        else:
            os.system('clear')
        print(to_print)
        print('\n')
        prints = ['_', '1', '2']
        print('   ',end='')
        for i in range(self.board.board.shape[1]):
            print('  #'+str(i), end='  ')
        c_cnt = 0
        print('\n')
        for r in self.board.board:
            print('#'+ str(c_cnt) + ' ' * (2 - len(str(c_cnt))),end='')
            for c in r:
                print('  ',prints[c], end='  ')
            print('\n')
            c_cnt += 1

    def collect_hists(self, turn):
        if self.collect_ai_hists:
            probs, board = self.players[turn].mcts.probs_board()
            self.hists_prob.append(probs)
            self.hists_board.append(board)

    def start(self, graphics=True):
        """
        Start a game
        @graphics:  whether or not to show UI. If this is a human-involved game, you'd better set it true.
        """
        if graphics: 
            self.graphics()
        turn = 0
        grid_states = [Board.GridState.p1, Board.GridState.p2]
        act = None
        stones = 0
        while True:
            if graphics: 
                print('Player',turn + 1,'s turn:')
            if self.ps[turn] == Game.Player.AI:
                self.players[turn].mcts.enemy_move = act

            if self.use_hists is None:
                act = self.players[turn].get_valid_action()
            else:
                try:
                    last_move = np.where(self.use_hists[self.hist_it,:,:,2]==1)
                    act0 = last_move[0][0]
                    act1 = last_move[1][0]
                    # argmax_this_move = 
                except:
                    # print
                    input("Replay over.  player " + str(turn + 1) + " wins , position: " + str(self.hist_it))
                    self.winner = turn
                    return self.hist_it
                act = (act0, act1)
                while np.sum(self.use_hists[self.hist_it,:,:,:2]) == stones:
                    self.hist_it += 1
            if act is None:
                if graphics:
                    print('Player',turn + 1,'resigns')
                self.winner = 1 - turn
                break
            self.board.board[act[0]][act[1]] = grid_states[turn]
            stones += 1
            if graphics: 
                self.graphics('Last move of player' + str(turn+1) + ':' + str(act))
            termination = self.check_over(act, stones)
            if graphics and self.use_hists is not None:
                input()
                
            self.collect_hists(turn)
            # Check whether the game is over 
            if termination != Termination.going:
                if termination == Termination.won:
                    if graphics: 
                        print(grid_states[turn], 'won')
                    self.winner = turn
                else:
                    self.winner = None
                    if graphics: 
                        print('tie')
                break
            turn = 1 - turn

    def ai_hists(self):
        """
        return 
        [
            MCTS search probabilities(rows*cols), 
            evaluation board(rows*cols*4),
            winner(int scalar)
        ]
        """
        return [self.hists_prob, self.hists_board, self.winner] if self.collect_ai_hists else None

    def check_over(self, pos, stones):
        return check_over_full(self.board.board, pos, self.n_in_row, stones=stones)


def eval_mcts(rows, cols, n_in_row, mcts1, mcts2, verbose=True, sim_times=100, collect_ai_hists=False):
    """
    sim_times can be either a scalar or a two-element iterable, indicating the first or second player's first-move games
    """
    if isinstance(sim_times, int):
        sim_times = [sim_times, sim_times]
    player1_wincnt = 0
    player2_wincnt = 0
    #inh1 = True
    ai_hists = []
    for id in range(2):
        for i in range(sim_times[id]):
            game = Game(rows,cols,n_in_row,Game.Player.AI,Game.Player.AI, collect_ai_hists=collect_ai_hists)
            game.players[id].mcts.from_another_mcts(mcts1)
            game.players[1-id].mcts.from_another_mcts(mcts2)
            game.start(graphics=False)
            player1_wincnt += (game.winner == id)
            player2_wincnt += (game.winner == 1-id)
            if verbose:
                print(player1_wincnt, player2_wincnt)
            if collect_ai_hists:
                ai_hists.append(game.ai_hists())
    sim_times = sum(sim_times)
    player1_wincnt /= sim_times
    player2_wincnt /= sim_times
    tie_rate = 1.0 - player1_wincnt - player2_wincnt
    return [player1_wincnt, player2_wincnt, tie_rate, ai_hists]


def _test_replay():
    path = r'F:\Software\vspro\NInRow\NInRow\zero_nns115\godd_mod\nomcts\selfplay0.npy'
    # print(np.load(path).shape)
    ret = 0
    w1 = 0
    w2 = 0
    t = 0
    try:
        while True:
            game = Game(11,11,5,use_hists=path, hist_it=ret)
            ret = game.start()
            w1 += game.winner == 0
            w2 += game.winner == 1
            t += game.winner is None
    except:
        pass
    print(w1, w2, t)


if __name__=='__main__':
    _test_replay()