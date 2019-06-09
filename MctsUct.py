# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI-AlphaRow: MCTS of UCT algorithm's implementation, class encapulated
"""


import numpy as np
from enum import IntEnum
import game_utils


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY

class Node:
    def __init__(self, parent=None, move=None, role=None, sim_board=None, mcts=None):
        self.visits = 0
        self.value = 0.0
        self.role = role
        self.mcts = mcts
        self.parent = parent
        self.move = move
        self.move_sim_board(sim_board)
        self.won = self.mcts.check_over(sim_board, move) if move is not None else game_utils.Termination.going
        self.childrens = []
        self.avails = self.init_avails() #empty grids
        self.unvisited = self.avails.copy()

    def search_child_move(self, move):
        for child in self.childrens:
            if child.move == move:
                return child
        return None

    def init_avails(self):
        avails = []
        if self.parent is not None:
            avails = self.parent.avails.copy()
            avails.remove(self.move)
        else:
            for r in range(self.sim_board.shape[0]):
                for c in range(self.sim_board.shape[1]):
                    if self.sim_board[r][c] == Grid.GRID_EMP:
                        avails.append((r,c))
        return avails

    def move_sim_board(self, sim_board):
        if self.move is not None:
            sim_board[self.move[0]][self.move[1]] = self.role
        self.sim_board = sim_board

    def select(self):
        # return whether expanded
        if len(self.unvisited) != 0:
            return self.expand(), True
        elif self.won != game_utils.Termination.going:
            return self, False
        best = max(self.childrens, key=lambda child: \
                    child.value/(child.visits+0.01)+self.mcts.c*np.sqrt( np.log(self.visits+1.1)/ (child.visits+0.01)))
        best.move_sim_board(self.sim_board)
        return best, False

    def expand(self):
        selected = self.unvisited[-1]
        self.unvisited.pop()
        self.childrens.append(Node(self, selected, -self.role, self.sim_board,self.mcts))
        return self.childrens[-1]


class MctsUct:
    def __init__(self, board_rows_, board_cols_, n_target_=5, max_t_=5, max_acts_=1000):
        self.n_target = n_target_
        self.max_t = max_t_
        self.max_acts = max_acts_
        self.board_cols = board_cols_
        self.board_rows = board_rows_
        self.c = 1.0
        self.last_best = None
        self.inherit = True
    
    def set_inherit(self, inherit):
        self.inherit = inherit

    def set_max_act(self, max_acts_):
        self.max_acts = max_acts_

    def tree_policy(self, root):
        expanded = False
        selected = root
        while not expanded:
            if selected.won != game_utils.Termination.going:
                return selected
            selected, expanded = selected.select()
        return selected

    def default_policy(self, node):
        bd = node.sim_board
        avails = node.avails
        np.random.shuffle(avails)
        role = -node.role
        for avail in avails:
            bd[avail[0]][avail[1]] = role
            termination = self.check_over(bd, avail)
            if termination != game_utils.Termination.going:
                # print(bd, role, n_target)
                return role if game_utils.Termination.won == termination else None
            role = -role
        return None

    def check_over(self, bd, pos):
        return game_utils.Game.check_over_full(bd, pos, self.n_target)

    def backup(self, node, win_role):
        while node.parent != None:
            if win_role is None:
                node.visits += 1
                node = node.parent
                continue
            if win_role == node.role:
                node.value += 1
            #else:
                #node.value -= 1
            node.visits += 1
            node = node.parent

    def simulate(self, board):
        # print(board)
        if self.last_best is None or self.inherit:
            root = Node(None, None, Grid.GRID_ENY, board.copy(), self)
        else:
            enemy_move = np.where(self.last_best.sim_board != board)
            enemy_move = (enemy_move[0][0], enemy_move[1][0])
            root = self.last_best.search_child_move(enemy_move)
            if root is None: root = Node(None, None, Grid.GRID_ENY, board.copy(), self)
            # else: print("Get ",enemy_move)
        win_cnt = 0
        tie_cnt = 0
        t_sim = 0
        for new_board in np.repeat(board.reshape((1,)+board.shape),self.max_acts,0):
            # print('\n')
            if t_sim % 2000 == 0 and t_sim != 0:
                print('win rate:', win_cnt/t_sim, '  tie rate=', tie_cnt/t_sim)
            t_sim += 1
            root.sim_board = new_board
            leaf = self.tree_policy(root)
            if leaf.won != game_utils.Termination.going:
                win_cnt += (leaf.role == Grid.GRID_SEL and leaf.won == game_utils.Termination.won)
                tie_cnt += leaf.won == game_utils.Termination.tie
                self.backup(leaf, None if leaf.won == game_utils.Termination.tie else leaf.role)
                continue
            win_role = self.default_policy(leaf)
            if win_role != 0:
                self.backup(leaf, win_role)
                win_cnt += (win_role == Grid.GRID_SEL)
        return root

    def select_action(self, board):
        root = self.simulate(board.copy())
        root.sim_board = board.copy()
        best, _ = root.select()
        # print(best.sim_board)
        self.last_best = best
        return best.move


