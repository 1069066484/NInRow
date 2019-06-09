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
12 88 
game.players[0].set_strength(10)
game.players[1].set_strength(50)
game.players[1].mcts.set_inherit(False)

50: 28 20 / 21 28 / 20 27 / 31 17
game.players[0].set_strength(50)
game.players[1].set_strength(100)
game.players[1].mcts.set_inherit(True)

50: 29 17 / 39 10 / 30 15 / 37 12
game.players[0].set_strength(50)
game.players[1].set_strength(100)
game.players[1].mcts.set_inherit(False)
"""


if __name__=='__main__':
    main_debug()