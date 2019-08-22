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
from trainer.ZeroNNTrainerMP import *
def main_debug():
    MctsPuct.CHECK_DETAILS = True
    collect_ai_hists = False
    gb = '115'
    if True:
        path = join(mkdir(join(FOLDER_ZERO_NNS + gb, 'replays')), curr_time_str() + '_AI_test')

        game = Game(int(gb[0]) + 10,int(gb[1]) + 10,5, Game.Player.AI, Game.Player.human, collect_ai_hists=collect_ai_hists)
        zeroNN1 = ZeroNN(path=join(FOLDER_ZERO_NNS + gb, 'NNs'), 
                             ckpt_idx=join(FOLDER_ZERO_NNS + gb, 'NNs/model.ckpt-316206') )
        # model.ckpt-16533
        # zeroNN1 = None
        noise = 0
        if True:
            game.players[0].mcts.zeroNN = zeroNN1
            game.players[0].mcts.max_acts = 2048
        # game.players[0].mcts.val_mult = 2
        # game.players[0].mcts.noise = noise
        # game.players[0].mcts.do_policy_prune = False
        # game.players[0].mcts.hand_val = 0.5
        # game.players[0].c = 15


        # game.players[0].mcts.red_child = True
        # 10.8 9.5
        # game.players[0].mcts.further_check = False
        if False:
            game.players[1].mcts.zeroNN = zeroNN1
            game.players[1].mcts.max_acts = 1024
            game.players[1].mcts.noise = noise

        game.start(graphics=True)

        pk.dump(game.acts, open(pkfn(path), 'wb'))
    input("over")

    if collect_ai_hists:
        debug = mkdir("debug")
        probs, eval_board, winner = game.ai_hists()
        sp0, sp1, sp2 = ZeroNNTrainer.hists2enhanced_train_data([[probs, eval_board, winner] ])
        np.save(join(debug, npfn('sp0')), sp0)
        np.save(join(debug, npfn('sp1')), sp1)
        np.save(join(debug, npfn('sp2')), sp2)
        exit()
        print(len(probs), len(eval_board), winner)
        print(probs[0].shape, eval_board[0].shape, winner)
        game = Game(11,11,5,use_hists=np.array(eval_board))
        game.start()
    

def test_664():
    winner = 0
    while True:
        MctsPuct.CHECK_DETAILS = True
        game = Game(5,5,4, Game.Player.AI, Game.Player.AI, collect_ai_hists=False)
        zeroNN1 = ZeroNN(path=r'F:\Software\vspro\NInRow\NInRow\test554\NNs', ckpt_idx=r'F:\Software\vspro\NInRow\NInRow\test554\NNs\model.ckpt-257210')
        #216633
                                     #ckpt_idx=r'F:\Software\vspro\NInRow\NInRow\test554\NNs\model.ckpt-177628')
        """
         [[0.01 0.01 0.   0.   0.  ]
         [0.   0.01 0.01 0.02 0.01]
         [0.01 0.06 0.17 0.24 0.03]
         [0.   0.02 0.03 0.15 0.06]
         [0.   0.02 0.06 0.07 0.03]]
        """
        noise = 0

        game.players[0].mcts.zeroNN = zeroNN1
        game.players[1].mcts.zeroNN = zeroNN1
        
        game.players[0].mcts.max_acts = 256
        game.players[1].mcts.max_acts = 256

        game.players[0].mcts.noise = noise
        game.players[1].mcts.noise = noise

        game.start(graphics=True)
        winner += (game.winner == 0) * 2 - 1
        input(str(winner))


def eval_debug():
    # zeroNN1 = ZeroNN(verbose=False,path=join(FOLDER_ZERO_NNS, 'NNs'))
    zeroNN1 = ZeroNN(path=join(FOLDER_ZERO_NNS + '115', 'NNs'), 
                     ckpt_idx=join(FOLDER_ZERO_NNS + '115', 'NNs/model.ckpt-145981'))
    # zeroNN2 = ZeroNN(verbose=False,path=join(FOLDER_ZERO_NNS, 'NNs'))
    # zeroNN1 = None
    zeroNN2 = ZeroNN(path=join(FOLDER_ZERO_NNS + '115', 'NNs'), 
                     ckpt_idx=join(FOLDER_ZERO_NNS + '115', 'NNs/model.ckpt-105633'))
    zeroNN2 = None
    mcts1 = Mcts(0,0,zeroNN=zeroNN1,max_acts_=512)
    mcts2 = Mcts(0,0,zeroNN=zeroNN2,max_acts_=1024)
    winrate1, winrate2, tie_rate, ai_hists = \
        eval_mcts(11, 11, 5, mcts1, mcts2, sim_times=5, verbose=True)
    print(winrate1, winrate2, tie_rate)
    '''
    First generation(786) VS Zero generation(0) - 512: 0.775 0.225 0.0 (40 games)
    Second generationA(8088) VS Zero generation(0) - 512: 0.95 0.0 0.05  (40 games)
    Second generationA(8088) VS First generation(786) - 512: 0.8 0.15 0.05  (20 games)
    Thrid generation(22730) VS Second generationA(8088) - 512: 0.55 0.35 0.1  (20 games)
    Thrid generation(22730) VS Zero generation(0) - 512: 0.95 0.05 0.0  (20 games)
    Fourth generation(56017) VS Thrid generation(22730) - 256: 0.6 0.35 0.05 (20 games)
    Fifth generation(61321) VS Fourth generation(56017) - 256: 0.65 0.35 0.0 (20 games)
    Fifth generation(61321) VS Zero generation(0) - 256: 0.7 0.0 0.3 (20 games)
    Sixth generation(69028)
    Seventh generation(69677)
    Eighth generation(70674) VS Fifth generation(61321)  - 256: 0.65 0.35 0.0 (20 games)
    Ninth generation(73799) VS Eighth generation(70674)  - 256: 0.65 0.35 0.0 (20 games)
    Tenth generation(74240) VS Ninth generation(73799)  - 256: 0.7 0.3 0.0 (20 games)
    Tenth generation(74240) VS Fifth generation(61321)  - 256: 0.6 0.4 0.0 (20 games)
    Eleventh generation(77702) VS Tenth generation(74240)  - 256: 0.5 0.45 0.05 (20 games)
    99830 VS 77702 0.65 0.35 0.0(20 games)

    '''


if __name__=='__main__':
    main_debug()
    # eval_debug()
    # test_664()
    """
    hists[i] is history of one single game
        hists[i][0]: a list of r*c arrays indicating probs
        hists[i][1]: a list of r*c*4 arrays indicating eval boards
        hists[i][2]: the win role
    """

