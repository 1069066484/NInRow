from random import choice, shuffle
from math import log, sqrt
import numpy as np
from enum import IntEnum
import MctsUct
import os
import sys


over_funs = [
    lambda i,p:(p[0]+i,p[i]+i),
    lambda i,p:(p[0]-i,p[i]+i),
    lambda i,p:(p[0],p[i]+i),
    lambda i,p:(p[0]+i,p[i])
    ]


class Termination(IntEnum):
    going = 100
    won = 101
    tie = 102


class AI:
    def __init__(self, board, n_in_row, max_acts):
        self.board = board.board
        self.mcts = MctsUct.MctsUct(self.board.shape[0], self.board.shape[1], n_in_row, 5, max_acts)

    def set_id(self, grid_id):
        self.grid_id = grid_id
        trans_dict = {Board.GridState.empty:MctsUct.Grid.GRID_EMP}
        if grid_id == Board.GridState.p1:
            trans_dict.update({Board.GridState.p1: MctsUct.Grid.GRID_SEL, Board.GridState.p2: MctsUct.Grid.GRID_ENY})
        else:
            trans_dict.update({Board.GridState.p2: MctsUct.Grid.GRID_SEL, Board.GridState.p1: MctsUct.Grid.GRID_ENY})
        self.trans_dict = trans_dict

    def set_strength(self, strength):
        self.mcts.set_max_act(strength)
        # print(self.mcts.max_acts)

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
    def __init__(self, board, n_in_row, max_acts):
        self.board = board

    def set_id(self, grid_id):
        pass

    def set_strength(self, strength):
        pass

    def get_valid_action(self):
        while True:
            try:
                s = input("Your move: ")
                location = [int(s[0]), int(s[2])]
                if self.board.get_loc_state(location) != Board.GridState.empty:
                    raise Exception()
                return location
            except:
                print("invalid move")


class Game:
    class Player(IntEnum):
        AI = 0
        human = 1

    def __init__(self, rows, cols, n_in_row, p1=Player.human, p2=Player.AI):
        self.board = Board(rows, cols)
        init_player = [AI, Human]
        self.players = [init_player[p](self.board, n_in_row, 20000) for p in [p1, p2]]
        self.players[0].set_id(Board.GridState.p1)
        self.players[1].set_id(Board.GridState.p2)
        self.n_in_row = n_in_row
        assert(self.n_in_row <= rows and self.n_in_row <= cols )

    def graphics(self, to_print=''):
        os.system('cls')
        print(to_print)
        print('\n')
        prints = ['_', '1', '2']
        print('  ',end='')
        for i in range(self.board.board.shape[1]):
            print('   #'+str(i), end='   ')
        c_cnt = 0
        print('\n')
        for r in self.board.board:
            print('#'+ str(c_cnt),end='')
            for c in r:
                print('   ',prints[c], end='   ')
            print('\n\n')
            c_cnt += 1

    def start(self, graphics=True):
        if graphics: self.graphics()
        turn = 0
        grid_states = [Board.GridState.p1, Board.GridState.p2]
        while True:
            if graphics: print('Player',turn + 1,'s turn:')
            act = self.players[turn].get_valid_action()
            self.board.board[act[0]][act[1]] = grid_states[turn]
            if graphics: self.graphics('Last move of player' + str(turn+1) + ':' + str(act))
            termination = self.check_over(act)
            if termination != Termination.going:
                if termination == Termination.won:
                    if graphics: print(grid_states[turn], 'won')
                    self.winner = turn
                else:
                    self.winner = None
                    if graphics: print('tie')
                break
            turn = (turn + 1) % 2

    def check_over(self, pos):
        return Game.check_over_full(self.board.board, pos, self.n_in_row)

    @staticmethod
    def check_over_full(board, pos, targets):
        """
        return Termination
        """
        def is_pos_legal(pos):
            return board.shape[0] > pos[0] >= 0 and board.shape[1] > pos[1] >= 0
        bd = board
        for f in over_funs:
            role = bd[pos[0]][pos[1]]
            score = 0
            pos_t = pos
            while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
                score += 1
                pos_t = f(1,pos_t)
            pos_t = f(-1,pos)
            while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
                score += 1
                pos_t = f(-1,pos_t)
            if score >= targets:
                return Termination.won
        return Termination.going if np.sum(bd != MctsUct.Grid.GRID_EMP) != bd.size else Termination.tie

