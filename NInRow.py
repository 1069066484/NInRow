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
from games.game import *
from mcts import MctsPuct


MctsPuct.CHECK_DETAILS = False


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


from nets.ZeroNN import *
def main_debug():
    if True:
        game = Game(11,11,5, Game.Player.AI, Game.Player.AI, collect_ai_hists=True)
        zeroNN1 = ZeroNN(path=join(FOLDER_ZERO_NNS + '115', 'NNs'), 
                             ckpt_idx=-1)#join(FOLDER_ZERO_NNS + '885', 'NNs/model.ckpt-10300') )
        zeroNN1 = None
    
        game.players[0].mcts.zeroNN = zeroNN1
        game.players[0].mcts.max_acts = 512
        # game.players[0].mcts.hand_val = 0.5
        # game.players[0].c = 15


        # game.players[0].mcts.red_child = True
        # 10.8 9.5
        # game.players[0].mcts.further_check = False
    
        game.players[1].mcts.zeroNN = zeroNN1
        game.players[1].mcts.max_acts = 512
        game.players[1].mcts.hand_val = 0

        game.start(graphics=True)
    input("over")
    # return None
    probs, eval_board, winner = game.ai_hists()
    print(len(probs), len(eval_board), winner)
    print(probs[0].shape, eval_board[0].shape, winner)
    input()
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


def eval_debug():
    # zeroNN1 = ZeroNN(verbose=False,path=join(FOLDER_ZERO_NNS, 'NNs'))
    zeroNN1 = ZeroNN(path=join(FOLDER_ZERO_NNS + '115', 'NNs'))
    # zeroNN2 = ZeroNN(verbose=False,path=join(FOLDER_ZERO_NNS, 'NNs'))
    zeroNN2 = None
    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=256)
    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=256)
    winrate1, winrate2, tie_rate, ai_hists = \
        eval_mcts(11, 11, 5, mcts1, mcts2, sim_times=10, verbose=True)
    print(winrate1, winrate2, tie_rate)
    '''
    
    '''


if __name__=='__main__':
    main_debug()
    # eval_debug()
    """
    hists[i] is history of one single game
        hists[i][0]: a list of r*c arrays indicating probs
        hists[i][1]: a list of r*c*4 arrays indicating eval boards
        hists[i][2]: the win role
    """