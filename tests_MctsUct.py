
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
"""
MctsUct
inherit: True False
  w:     0.87 0.12
"""


def c_test():
    c1 = 2.0
    for c2 in [4.0,2.5,1.5,1.0,0.5]:
        mcts1 = Mcts(0,0,0,5,200,c=c1)
        mcts2 = Mcts(0,0,0,5,200,c=c2)
        player1_winprob, player2_winprob = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=False)
        print('\nc:',mcts1.c, mcts2.c,'\n',
              'w:', player1_winprob,player2_winprob)
"""
MctsUct
c: 1.0 10.0
 w: 0.88 0.11

c: 1.0 5.0
 w: 0.74 0.25

c: 1.0 2.0
 w: 0.42 0.53

c: 1.0 0.5
 w: 0.47 0.49

c: 1.0 0.1
 w: 0.48 0.47

 
c: 2.0 4.0
 w: 0.67 0.31

c: 2.0 2.5
 w: 0.57 0.41

c: 2.0 1.5
 w: 0.47 0.5

c: 2.0 1.0
 w: 0.44 0.52

c: 2.0 0.5
 w: 0.45 0.53
"""

def penelty_test():
    penelty1 = 0.0
    for penelty2 in [0.0,0.2,0.5]:
        mcts1 = Mcts(0,0,0,5,200,penelty=penelty1)
        mcts2 = Mcts(0,0,0,5,200,penelty=penelty2)
        player1_winprob, player2_winprob = eval_mcts(5,5,4,mcts1,mcts2,sim_times=500, verbose=False)
        print('\nc:',mcts1.penelty, mcts2.penelty,'\n',
              'w:', player1_winprob,player2_winprob)
"""
MctsUct
c: 0.5 0.0
 w: 0.54 0.46

c: 0.5 0.1
 w: 0.47 0.52

c: 0.5 0.2
 w: 0.5 0.48

c: 0.5 0.5
 w: 0.43 0.53

c: 0.5 0.7
 w: 0.43 0.5

c: 0.5 1.0
 w: 0.4 0.56
 

c: 0.0 0.0
 w: 0.475 0.525

c: 0.0 0.05
 w: 0.525 0.475

c: 0.0 0.1
 w: 0.4975 0.4975

c: 0.0 0.2
 w: 0.44 0.5575

c: 0.0 0.5
 w: 0.465 0.5325

c: 0.0 1.0
 w: 0.4825 0.5125

 c: 0.0 0.0
 w: 0.536 0.464

c: 0.0 0.2
 w: 0.443 0.554

c: 0.0 0.5
 w: 0.441 0.555
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