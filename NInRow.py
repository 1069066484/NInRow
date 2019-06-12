# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: entrace script.
"""

from random import choice, shuffle
from math import log, sqrt
import numpy as np
from enum import IntEnum
import os
import sys
from game_utils import *


def main_exe():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--player1", help="role of player1: 0 for AI or 1 for human", type=int, required=True)
    parser.add_argument("-p2", "--player2", help="role of player1: 0 for AI or 1 for human", type=int, required=True)
    parser.add_argument("-d1", "--difficulty1", help="difficulty of AI1, ranging from 1000-INF, work only if player1 is AI",
                        type=int, default=20000)
    parser.add_argument("-d2", "--difficulty2", help="difficulty of AI2, ranging from 1000-INF, work only if player2 is AI",
                        type=int, default=20000)
    args, _ = parser.parse_known_args(sys.argv[1:])
    game = Game(5,5,4,Game.Player(args.player1),Game.Player(args.player2))
    game.players[0].set_strength(args.difficulty1)
    game.players[1].set_strength(args.difficulty2)
    game.start()
    input("Press any key to exit")


def main_debug():
    game = Game(3,3,3,Game.Player.AI,Game.Player.AI,collect_ai_hists=True)
    game.players[0].mcts.max_acts = 1000
    game.players[1].mcts.max_acts = 100
    game.start(graphics=True)
    probs, eval_board, winner = game.ai_hists()
    for i in range(3):
        print(probs[i])
        print(eval_board[i][:,:,0])
        print(eval_board[i][:,:,1])
        print(eval_board[i][:,:,2])
        print(eval_board[i][:,:,3])
        print('\n\n')
    print(np.array(probs).shape)
    print(np.array(eval_board).shape)
    print(winner)


def reversed_eval_board(board):
    board_new = board.copy()
    # the oppenent is the next to move
    board_new[:,:,3] = 1 - board[:,:,3]
    # swap the player to move
    board_new[:,:,0] = board[:,:,1]
    board_new[:,:,1] = board[:,:,0]
    return board_new

def hists2enhanced_train_data(ai_hists):
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
    return [np.array(X, dtype=np.int8), np.array(Y_policy), np.array(Y_value)]


def hists_test():
    mcts1 = Mcts(0,0,0)
    mcts2 = Mcts(0,0,0)
    player1_winprob, player2_winprob, tie_rate, hists = eval_mcts(3,3,3,mcts1,mcts2,sim_times=1, verbose=False, collect_ai_hists=True)
    hists = hists2enhanced_train_data(hists)
    print(hists[0].shape, hists[1].shape, hists[2].shape)


if __name__=='__main__':
    hists_test()
    """
    hists[i] is history of one single game
        hists[i][0]: a list of r*c arrays indicating probs
        hists[i][1]: a list of r*c*4 arrays indicating eval boards
        hists[i][2]: the win role
    """