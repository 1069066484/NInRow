# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: definitions related to termination.
"""

from enum import IntEnum
import numpy as np


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY


class Termination(IntEnum):
    going = 100
    won = 101
    tie = 102


move_funs = [
    lambda i,p:(p[0]+i,p[i]+i),
    lambda i,p:(p[0]-i,p[i]+i),
    lambda i,p:(p[0],p[i]+i),
    lambda i,p:(p[0]+i,p[i])
    ]


def check_over_full(board, pos, targets, ret_val=False):
    """
    return Termination
    An efficient algorithm is used to check the game is over or not.
    Time complexity os O(S), where S is the board size.
    """
    def is_pos_legal(pos):
        return board.shape[0] > pos[0] >= 0 and board.shape[1] > pos[1] >= 0
    bd = board
    val = 0
    for f in move_funs:
        role = bd[pos[0]][pos[1]]
        score = 0
        margins = 0
        pos_t = pos
        while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
            score += 1
            pos_t = f(1,pos_t)
        if is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == Grid.GRID_EMP:
            margins += 1
        pos_t = f(-1,pos)
        while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
            score += 1
            pos_t = f(-1,pos_t)
        if is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == Grid.GRID_EMP:
            margins += 1
        if score >= targets:
            return [Termination.won, 1] if ret_val else Termination.won
        if score + margins >= targets:
            val = max(val, (score + margins - targets + 1) * 0.5 - (margins == 1) * 0.4)
    for r in bd:
        for g in r:
            if g == Grid.GRID_EMP:
                return [Termination.going, val] if ret_val else Termination.going
    return [Termination.tie, 0] if ret_val else Termination.tie


def test_check_over_full():
    print(
        check_over_full(
            np.array(
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,1,2,0,0],
         [0,0,2,0,0],
         [0,0,0,0,0],]),
         [2,2],
         4,
         True
        )
        )


if __name__=='__main__':
    test_check_over_full()

