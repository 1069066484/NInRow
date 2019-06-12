# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: MCTS of PUCT algorithm's implementation. Difference between UCT and PUCT is that
        PUCT uses a P as prior when selecting a child node.
        The important differences can be seen in function the Node::select.
"""
import numpy as np
from enum import IntEnum
import game_utils
from ZeroNN import ZeroNN
#from CNN import CNN


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY


class Node:
    def __init__(self, parent=None, move=None, role=None, sim_board=None, mcts=None, p=1):
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
        self.p = p 
        
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
        if self.won != game_utils.Termination.going:
            return self, False
        elif len(self.childrens) == 0:
            self.expand()
            return self, True
        best = max(self.childrens, key=lambda child: \
                    child.value/(child.visits+0.01)+        # Q
                   self.mcts.c*child.p*np.sqrt(self.visits)/ (child.visits+1.0))    # U
        best.move_sim_board(self.sim_board)
        return best, False

    def create_eval_board(self):
        """
        Four channels:
            1. self pieces
            2. opponent pieces
            3. the node's movement(last movement)
            4. current player to move
        The board can be reversed in terms of the opponent's view
        """
        eval_board = np.zeros(shape=[self.sim_board.shape[0],self.sim_board.shape[1], 4], dtype=np.int8)
        eval_board[:,:,0][self.sim_board==self.role] = 1
        eval_board[:,:,1][self.sim_board==-self.role] = 1
        if self.move is not None:
            eval_board[:,:,2][self.move[0]][self.move[1]] = 1

        # at this node, the self.role has moved, so the current player to move is the oppenent
        eval_board[:,:,3][:,:] = self.role != Grid.GRID_SEL
        return eval_board

    def probs(self):
        probs = np.zeros(self.sim_board.shape)
        for child in self.childrens:
            probs[child.move[0]][child.move[1]] = child.visits
        return probs / self.visits

    def expand(self):
        """
        We expand all children
        """
        self.value, probs = self.mcts.eval_state(self.create_eval_board())
        for r,c in self.avails:
            self.childrens.append(Node(self, (r,c), -self.role, self.sim_board.copy(), self.mcts, probs[r][c]))


class MctsPuct:
    def __init__(self, board_rows_, board_cols_, n_target_=5, max_t_=5,
                 max_acts_=1000, c=4.0,inherit=True,penelty=0.5,fix_p=1, zeroNN=None):
        self.n_target = n_target_
        self.max_t = max_t_
        self.max_acts = max_acts_
        self.board_cols = board_cols_
        self.board_rows = board_rows_
        self.c = c
        self.last_best = None
        self.inherit = inherit
        self.fix_p = fix_p
        self.zeroNN = zeroNN
        self.enemy_move = None

    def from_another_mcts(self, other):
        self.max_t = other.max_t
        self.max_acts = other.max_acts
        self.c = other.c
        self.inherit = other.inherit
        self.fix_p = other.fix_p
        self.zeroNN = other.zeroNN

    def tree_policy(self, root):
        expanded = False
        selected = root
        while not expanded:
            if selected.won != game_utils.Termination.going:
                return selected
            selected, expanded = selected.select()
        return selected

    def default_policy(self, node):
        """
        MctsPuct does not use default_policy according to alphaGo Zero
        """
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

    def backup(self, node, win_rate):
        """
        If win_role is GRID_SEL, win_rate should have the same sign as GRID_SEL
        If win_role is GRID_ENY, win_rate should have the same sign as GRID_ENY
        The trick can simplify the backup process
        """
        while node != None:
            node.value += node.role * win_rate
            node.visits += 1
            node = node.parent

    def simulate(self, board):
        enemy_move = self.enemy_move
        if self.last_best is None or self.inherit:
            root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        else:
            root = self.last_best.search_child_move(enemy_move)
            if root is None: root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        win_cnt = 0
        tie_cnt = 0
        t_sim = 0
        for new_board in np.repeat(board.reshape((1,)+board.shape),self.max_acts,0):
            # print('\n')
            # if t_sim % (self.max_acts//5) == 0 and t_sim != 0:
            #     print('win rate:', win_cnt/t_sim, '  tie rate=', tie_cnt/t_sim)
            t_sim += 1
            root.sim_board = new_board
            leaf = self.tree_policy(root)
            if leaf.won != game_utils.Termination.going:
                win_cnt += (leaf.role == Grid.GRID_SEL and leaf.won == game_utils.Termination.won)
                tie_cnt += leaf.won == game_utils.Termination.tie
                self.backup(leaf, 0.0 if leaf.won == game_utils.Termination.tie else leaf.value)
            # No default policy
        return root

    def probs_board(self):
        return [self.root.probs(), self.root.create_eval_board()]

    def select_action(self, board):
        self.root = self.simulate(board.copy())
        self.root.sim_board = board.copy()
        best, _ = self.root.select()
        self.root.sim_board = board.copy()
        self.last_best = best
        return best.move

    def probs_board(self):
        return [self.root.probs(), self.root.create_eval_board()]

    def eval_state(self, board):
        """
        Input board is a 4*board_rows*board_rows matrix
        Use NN to evaluate the current game state, return a double-element list [value, policy]
            value is a scalar while policy is a board_rows*board_cols matrix
        """
        value, policy = self.zeroNN.predict(board.reshape([1]+list(board.shape)))
        return value[0][0], policy.reshape([-1] + list(board.shape[:-1]))[0]

