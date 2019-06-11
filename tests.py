
# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: hyper-parameter tests.
"""

from random import choice, shuffle
from math import log, sqrt
import numpy as np
from enum import IntEnum
import os
import sys
from game_utils import *


def inherit_test():
    inherit1 = True
    inherit2 = False
    mcts1 = Mcts(0,0,0,5,200,inherit=inherit1)
    mcts2 = Mcts(0,0,0,5,200,inherit=inherit2)
    player1_winprob, player2_winprob = eval_mcts(5,5,4,mcts1,mcts2,sim_times=150, verbose=False)
    print('\ninherit:',inherit1,inherit2,'\n',
            ' w:    ', player1_winprob,player2_winprob)


def c_test():
    c1 = 2.0
    for c2 in [4.0,2.5,1.5,1.0,0.5]:
        mcts1 = Mcts(0,0,0,5,200,c=c1)
        mcts2 = Mcts(0,0,0,5,200,c=c2)
        player1_winprob, player2_winprob = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=False)
        print('\nc:',mcts1.c, mcts2.c,'\n',
              'w:', player1_winprob,player2_winprob)


def penelty_test():
    penelty1 = 0.0
    for penelty2 in [0.0,0.2,0.5]:
        mcts1 = Mcts(0,0,0,5,200,penelty=penelty1)
        mcts2 = Mcts(0,0,0,5,200,penelty=penelty2)
        player1_winprob, player2_winprob = eval_mcts(5,5,4,mcts1,mcts2,sim_times=500, verbose=False)
        print('\nc:',mcts1.penelty, mcts2.penelty,'\n',
              'w:', player1_winprob,player2_winprob)


def p_test():
    player1_wincnt = 0
    player2_wincnt = 0
    #inh1 = True
    for i in range(500):
        game = Game(5,5,4,Game.Player.AI,Game.Player.AI)
        game.players[0].set_strength(100)
        game.players[1].set_strength(100)
        game.players[0].mcts.fix_p = 1
        game.players[1].mcts.fix_p = 1.0
        game.start(graphics=False)
        player1_wincnt += game.winner == 0
        player2_wincnt += game.winner == 1
        print(player1_wincnt, player2_wincnt)
    print(game.players[0].mcts.fix_p, game.players[1].mcts.fix_p)


if __name__=='__main__':
    penelty_test()