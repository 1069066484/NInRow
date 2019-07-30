
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
from games.game import *


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


def update_test():
    update_itv1 = 0
    update_itv2 = 16
    mcts1 = Mcts(0,0,0,5,256, update_itv=update_itv1)
    mcts2 = Mcts(0,0,0,5,256, update_itv=update_itv2)
    player1_winprob, player2_winprob, tieprob, _ = eval_mcts(6,6,4,mcts1,mcts2,sim_times=40, verbose=True)
    print('\n update_itv1:',update_itv1,update_itv2,'\n',
            ' w:    ', player1_winprob, player2_winprob, tieprob)
"""
 update_itv: 1 32
  w:     0.4375 0.5375 0.025000000000000022


"""


def split_test():
    mcts1 = Mcts(0,0,0,5,256, max_update=0, split=0)
    mcts2 = Mcts(0,0,0,5,256, max_update=0, split=2)
    player1_winprob, player2_winprob, tie, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=True)
    print('\n split:',0,2,'\n',
            ' w:    ', player1_winprob, player2_winprob, tie)


def nzprob_enh_test():
    mcts1 = Mcts(0,0,0,5,256, nzprob_enh=0)
    mcts2 = Mcts(0,0,0,5,256, nzprob_enh=2)
    player1_winprob, player2_winprob, tie, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=True)
    print('\n split:',0,2,'\n',
            ' w:    ', player1_winprob, player2_winprob, tie)



def further_check_test():
    mcts1 = Mcts(0,0,0,5,128, further_check=False)
    mcts2 = Mcts(0,0,0,5,128, further_check=True)
    player1_winprob, player2_winprob, tie, _ = eval_mcts(6,6,4,mcts1,mcts2,sim_times=20, verbose=True)
    print('\n further_check:',mcts1.further_check,mcts2.further_check,'\n',
            ' w:    ', player1_winprob, player2_winprob, tie)
"""
 further_check: False True
  w:     0.15 0.85 0.0
"""


def defpolicy_test():
    mcts1 = Mcts(0,0,0,5,128, usedef=[3,5])
    mcts2 = Mcts(0,0,0,5,128, usedef=None)
    player1_winprob, player2_winprob, tie, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=True)
    print('\n usedef:',mcts1.usedef,mcts2.usedef,'\n',
            ' w:    ', player1_winprob, player2_winprob, tie)
'''

 usedef: 3 None
  w:     0.3 0.43 0.26999999999999996

 usedef: 2 1
  w:     0.47 0.34 0.19

 usedef: 3 1
  w:     0.4 0.35 0.25

 usedef: 2 None
  w:     0.37 0.41 0.22000000000000003

 usedef: 1 None
  w:     0.35 0.39 0.26

 usedef: [2, 2] [None, 1]
  w:     0.36 0.42 0.22000000000000003
'''


def hand_val_test():
    mcts1 = Mcts(0,0,0,5,128, hand_val=0)
    mcts2 = Mcts(0,0,0,5,128, hand_val=0)
    player1_winprob, player2_winprob, tie, _ = eval_mcts(5,5,4,mcts1,mcts2,sim_times=50, verbose=True)
    print('\n hand_val:',mcts1.hand_val,mcts2.hand_val,'\n',
            ' w:    ', player1_winprob, player2_winprob, tie)
'''
 hand_val: 1 0
  w:     0.1 0.35 0.55
 hand_val: 0.5 0
  w:     0.13 0.5 0.37
 hand_val: 0.1 0
  w:     0.28 0.52 0.19999999999999996

'''



if __name__=='__main__':
    # further_check_test()
    # defpolicy_test()
    hand_val_test()