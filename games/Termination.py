# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: definitions related to termination.
"""

from enum import IntEnum
import numpy as np
import time


TERM_T = [0,0]


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY
GRID_EMP_INT = int(Grid.GRID_EMP)
GRID_ENY_INT = int(Grid.GRID_ENY)
GRID_SEL_INT = int(Grid.GRID_SEL)


class Termination(IntEnum):
    going = 100
    won = 101
    tie = 102


DECIDE_VAL = 0.1

move_funs = [
    lambda i,p:(p[0]+i,p[1]+i),
    lambda i,p:(p[0]-i,p[1]+i),
    lambda i,p:(p[0],p[1]+i),
    lambda i,p:(p[0]+i,p[1])
    ]


def check_over_full(board, pos, targets, ret_val=False, stones=None):
    """
    return Termination
    An efficient algorithm is used to check the game is over or not.
    Time complexity = targets * 8 if stones provided
    """
    is_pos_legal = lambda pos: board.shape[0] > pos[0] >= 0 and board.shape[1] > pos[1] >= 0
    
    bd = board
    val = 0
    bonus = 0

    role = bd[pos[0]][pos[1]]

    for f in move_funs:
        score = 1
        margins = 0
        pos_t = (pos[0], pos[1])

        for p in [1, -1]:
            
            pos_t = f(p, pos)
            while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
                score += 1
                pos_t = f(p,pos_t)

                # give bonus value = 2 if there is a self-grid around
                bonus += 2
            
            # empty
            if is_pos_legal(pos_t):
                # if this is an empty grid 
                if not bd[pos_t[0]][pos_t[1]]:
                    margins += 1
                    bonus += 1
                # if this is an enemy grid
                else:
                    bonus -= 1
            else:
                bonus -= 3

        if score >= targets:
            return [Termination.won, 1] if ret_val else Termination.won

        # There is no way to form an N-in-row pattern, add a panelty
        # (N-1)-in-row with space around
        if score == targets - 1 and margins == 2:
            val = 0.2
    # max(bonus) = 10
    # multiply 0.01 to set the maximum bonus to 0.1
    val += (bonus * 0.001)
    if stones is not None:
        if stones == bd.size:
            return [Termination.tie, 0] if ret_val else Termination.tie
        else:
            return [Termination.going, val] if ret_val else Termination.going
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
         [0,0,2,0,0],
         [0,0,2,0,0],
         [0,1,2,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],]),
         [2,2],
         5,
         True
        )
        )

def eff_test():
    bd1 =  np.array(
        [[0,0,0,0,0,0,0],
         [0,0,-1,0,0,0,0],
         [0,0,-1,1,0,0,0],
         [0,0,0,0,0,0,0],
         [0,1,-1,0,0,0,0],
         [0,0,0,0,0,0,0],
         [0,0,0,0,1,0,0],])
    bd2 =  np.array(
        [[0,0,0,0,0,1,0],
         [0,0,0,0,0,-1,0],
         [0,0,0,1,0,1,0],
         [0,1,-1,0,0,0,0],
         [0,1,-1,0,0,0,0],
         [0,0,0,1,0,0,0],
         [0,-1,0,0,0,1,0],])
    t = time.clock()
    c = 0
    a1 = 0
    b1 = 0
    for i in range(50000):
        a, b = check_over_full(bd1, [1,2], 5, True, 6)
        a1, b1 =  check_over_full(bd2, [2,3], 5, True, 4)
        c += a + b + a1 + b1
    print(time.clock() - t, c, TERM_T)


if __name__=='__main__':
    eff_test()

