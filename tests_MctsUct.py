
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
    player1_winprob, player2_winprob, _, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=200, verbose=False)
    print('\ninherit:',inherit1,inherit2,'\n',
            ' w:    ', player1_winprob,player2_winprob)
"""
MctsUct
inherit: True False
  w:     0.5325 0.44
"""


def c_test():
    c1 = 0.2
    for c2 in [0.5,0.2,0.1]:
        mcts1 = Mcts(0,0,0,5,200,c=c1)
        mcts2 = Mcts(0,0,0,5,200,c=c2)
        player1_winprob, player2_winprob, _, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=False)
        print('\nc:',mcts1.c, mcts2.c,'\n',
              'w:', player1_winprob,player2_winprob)
"""
MctsUct
c: 2.0 4.0
 w: 0.71 0.28

c: 2.0 2.5
 w: 0.52 0.48

c: 2.0 1.5
 w: 0.35 0.64

c: 2.0 1.0
 w: 0.25 0.75

c: 2.0 0.5
 w: 0.06 0.92

c: 0.5 2.0
 w: 0.9 0.1

c: 0.5 1.0
 w: 0.74 0.24

c: 0.5 0.5
 w: 0.55 0.42

c: 0.5 0.2
 w: 0.39 0.59

 c: 0.2 0.5
 w: 0.68 0.32

c: 0.2 0.2
 w: 0.45 0.51

c: 0.2 0.1
 w: 0.57 0.37
 """

def penelty_test():
    penelty1 = 0.5
    for penelty2 in [0.0,0.5,1.0]:
        mcts1 = Mcts(0,0,0,5,200,penelty=penelty1)
        mcts2 = Mcts(0,0,0,5,200,penelty=penelty2)
        player1_winprob, player2_winprob, _, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=100, verbose=False)
        print('\npe:',mcts1.penelty, mcts2.penelty,'\n',
              'w:', player1_winprob,player2_winprob)
"""
c: 0.0 0.0
 w: 0.51 0.49

c: 0.0 0.2
 w: 0.45 0.54

c: 0.0 0.5
 w: 0.41 0.585

 pe: 0.5 0.0
 w: 0.53 0.47

pe: 0.5 0.5
 w: 0.495 0.485

pe: 0.5 1.0
 w: 0.5 0.465
"""



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
"""
MctsUct
286 194 / 298 194 / 297 192 / 301 188
1 1

292 196 / 306 189
1 1.0

"""



if __name__=='__main__':
    penelty_test()