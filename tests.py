
# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI-AlphaRow: console interface and game board logit.
"""

from random import choice, shuffle
from math import log, sqrt
import numpy as np
from enum import IntEnum
import os
import sys
from game_utils import *


def inherit_test():
    player1_wincnt = 0
    player2_wincnt = 0
    #inh1 = True
    for i in range(50):
        game = Game(4,4,4,Game.Player.AI,Game.Player.AI)
        game.players[0].set_strength(50)
        game.players[1].set_strength(100)
        game.players[1].mcts.set_inherit(True)
        game.start(graphics=False)
        player1_wincnt += game.winner == 0
        player2_wincnt += game.winner == 1
        print(player1_wincnt, player2_wincnt)
    """
    50: 28 20 / 21 28 / 20 27 / 31 17  :  p1-100 p2-92 tie-8
    game.players[0].set_strength(50)
    game.players[1].set_strength(100)
    game.players[1].mcts.set_inherit(True)

    50: 29 17 / 39 10 / 30 15 / 37 12  :  p1-135 p2-50 tie-15
    game.players[0].set_strength(50)
    game.players[1].set_strength(100)
    game.players[1].mcts.set_inherit(False)
    """


def c_test():
    player1_wincnt = 0
    player2_wincnt = 0
    #inh1 = True
    for i in range(100):
        game = Game(4,4,4,Game.Player.AI,Game.Player.AI)
        game.players[0].set_strength(100)
        game.players[1].set_strength(100)
        game.players[0].mcts.c = 2
        game.players[1].mcts.c = 1.0
        game.start(graphics=False)
        player1_wincnt += game.winner == 0
        player2_wincnt += game.winner == 1
        print(player1_wincnt, player2_wincnt)
    print(game.players[0].mcts.c, game.players[1].mcts.c)
    """
29 69
5 1.0

45 52  
1.5 1.0

55 41 / 49 45 / 59 38
1.0 1.0

59 40 / 52 46 / 52 43 / 48 48
2 1.0

46 53
0.1 1.0

21 77
10 1.0
"""

def penelty_test():
    player1_wincnt = 0
    player2_wincnt = 0
    #inh1 = True
    for i in range(50):
        game = Game(4,4,4,Game.Player.AI,Game.Player.AI)
        game.players[0].set_strength(100)
        game.players[1].set_strength(100)
        game.players[0].mcts.penelty = 0.5
        game.players[1].mcts.penelty = 10.0
        game.start(graphics=False)
        player1_wincnt += game.winner == 0
        player2_wincnt += game.winner == 1
        print(player1_wincnt, player2_wincnt)
    print(game.players[0].mcts.penelty, game.players[1].mcts.penelty)
"""
24 23
0.0 0.0
18 27
0.0 0.1
5 39
0.0 0.5
8 3
0.5 0.5
7 4
0.5 1.0
39 2
0.5 0
8 34
0.0 1.0
7 29
0.0 2.0
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
286 194 / 298 194 / 297 192 / 301 188
1 1

292 196 / 306 189
1 1.0

"""

if __name__=='__main__':
    p_test()