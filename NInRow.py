# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI-AlphaRow: console interface and game board logit.
"""

from random import choice, shuffle
from math import log, sqrt
import numpy as np
from enum import IntEnum
import MCTS_UCT
import os
import sys


class AI:
    def __init__(self, board, n_in_row, max_acts):
        self.board = board.board
        MCTS_UCT.init(self.board.shape[0], self.board.shape[1], n_in_row, 5, max_acts)

    def set_id(self, grid_id):
        self.grid_id = grid_id
        trans_dict = {Board.GridState.empty:MCTS_UCT.GRID_EMP}
        if grid_id == Board.GridState.p1:
            trans_dict.update({Board.GridState.p1: MCTS_UCT.GRID_SEL, Board.GridState.p2: MCTS_UCT.GRID_ENY})
        else:
            trans_dict.update({Board.GridState.p2: MCTS_UCT.GRID_SEL, Board.GridState.p1: MCTS_UCT.GRID_ENY})
        self.trans_dict = trans_dict

    def set_strength(self, strength):
        MCTS_UCT.set_max_act(strength)

    def get_valid_action(self):
        board = self.board.copy()
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                board[r][c] = self.trans_dict[board[r][c]]
        return MCTS_UCT.select_action(board)


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

    def start(self):
        self.graphics()
        turn = 0
        grid_states = [Board.GridState.p1, Board.GridState.p2]
        while True:
            print('Player',turn + 1,'s turn:')
            act = self.players[turn].get_valid_action()
            self.board.board[act[0]][act[1]] = grid_states[turn]
            self.graphics('Last move of player' + str(turn+1) + ':' + str(act))
            termination = self.check_over(act)
            if termination != MCTS_UCT.Termination.going:
                if termination == MCTS_UCT.Termination.won:
                    print(grid_states[turn], 'won')
                else:
                    print('tie')
                break
            turn = (turn + 1) % 2

    def check_over(self, pos):
        return MCTS_UCT.check_over_full(self.board.board, pos, self.n_in_row)


def main_exe():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--player1", help="role of player1: 0 for AI or 1 for human", type=int, required=True)
    parser.add_argument("-p2", "--player2", help="role of player1: 0 for AI or 1 for human", type=int, required=True)
    parser.add_argument("-d1", "--difficulty1", help="difficulty of AI1, ranging from 1000-INF, work only if player1 is AI",
                        type=int, default=20000)
    parser.add_argument("-d2", "--difficulty2", help="difficulty of AI1, ranging from 1000-INF, work only if player2 is AI",
                        type=int, default=20000)
    args, _ = parser.parse_known_args(sys.argv[1:])
    game = Game(5,5,4,Game.Player(args.player1),Game.Player(args.player2))
    game.players[0].set_strength(args.difficulty1)
    game.players[1].set_strength(args.difficulty2)
    game.start()
    input("Press any key to exit")


def main_debug():
    game = Game(6,6,4,Game.Player.AI,Game.Player.human)
    game.players[0].set_strength(20000)
    game.players[1].set_strength(20000)
    game.start()


if __name__=='__main__':
    main_debug()