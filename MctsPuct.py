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
import time
import threading
from queue import Queue


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY


class SearchOver(Exception):
    def __init__(self):
        pass


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
        self.avails = self.init_avails(sim_board) #empty grids
        self.unvisited = self.avails.copy()
        self.p = p 
        
    def search_child_move(self, move):
        for child in self.childrens:
            if child.move == move:
                return child
        return None

    def init_avails(self, sim_board):
        avails = []
        if self.parent is not None:
            avails = self.parent.avails.copy()
            avails.remove(self.move)
        else:
            for r in range(sim_board.shape[0]):
                for c in range(sim_board.shape[1]):
                    if sim_board[r][c] == Grid.GRID_EMP:
                        avails.append((r,c))
        return avails

    def move_sim_board(self, sim_board):
        if self.move is not None:
            sim_board[self.move[0]][self.move[1]] = self.role

    def select(self, sim_board):
        # return whether expanded
        if self.won != game_utils.Termination.going:
            return self, False, sim_board
        elif len(self.childrens) == 0:
            self.expand(sim_board)
            return self, True, sim_board
        best = max(self.childrens, key=lambda child: \
                    child.value/(child.visits+0.01)+        # Q
                   self.mcts.c*child.p*np.sqrt(self.visits)/ (child.visits+1.0))    # U
        best.move_sim_board(sim_board)
        return best, False, sim_board

    def create_eval_board(self, sim_board):
        """
        Four channels:
            1. self pieces
            2. opponent pieces
            3. the node's movement(last movement)
            4. current player to move
        The board can be reversed in terms of the opponent's view
        """
        eval_board = np.zeros(shape=[sim_board.shape[0],sim_board.shape[1], 4], dtype=np.int8)
        eval_board[:,:,0][sim_board==self.role] = 1
        eval_board[:,:,1][sim_board==-self.role] = 1
        if self.move is not None:
            eval_board[:,:,2][self.move[0]][self.move[1]] = 1

        # at this node, the self.role has moved, so the current player to move is the oppenent
        eval_board[:,:,3][:,:] = self.role != Grid.GRID_SEL
        return eval_board

    def probs(self, sim_board):
        probs = np.zeros(sim_board.shape)
        for child in self.childrens:
            probs[child.move[0]][child.move[1]] = child.visits
        # act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return probs / (self.visits + 0.1)

    def expand(self, sim_board):
        """
        We expand all children
        """
        # print('expand called', id(self))
        rets = self.mcts.eval_state_lock(self.create_eval_board(sim_board))
        if not self.mcts.eval_event.wait(0.1):
            self.value, probs = self.mcts.eval_state_single(self.create_eval_board(sim_board))
        else:
            self.value, probs = rets
        for r,c in self.avails:
            self.childrens.append(Node(self, (r,c), -self.role, sim_board.copy(), self.mcts, probs[r][c]))


class MctsPuct:
    def __init__(self, board_rows_, board_cols_, n_target_=5, max_t_=5,
                 max_acts_=1000, c=4.0,inherit=True,penelty=0.5,fix_p=1, zeroNN=None, n_threads=2):
        """
        8: 2-3.5, 3-2.86, 4-2.65
        4: 2-2.3, 3-1.85, 4-1.88
        2: 2-2.0, 3-1.88, 4-1.67
        1: 2-2.5, 3-2.37
        """
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
        self.init_syncs(n_threads)

    def init_syncs(self, n_threads):
        self.eval_queues = []
        self.eval_results = [[] for _ in range(n_threads)]
        self.n_threads = n_threads
        self.threads = n_threads
        self.locks = [threading.Lock() for _ in range(n_threads)]
        self.eval_lock = threading.Lock()
        self.eval_cond = threading.Condition()
        self.eval_event = threading.Event()
        self.search_over = 0

    def from_another_mcts(self, other):
        self.max_t = other.max_t
        self.max_acts = other.max_acts
        self.c = other.c
        self.inherit = other.inherit
        self.fix_p = other.fix_p
        self.zeroNN = other.zeroNN

    def tree_policy(self, root, sim_board):
        expanded = False
        selected = root
        while not expanded:
            if selected.won != game_utils.Termination.going:
                return selected
            selected, expanded, sim_board = selected.select(sim_board)
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

    def simulate_thread_fun(self, root, board, num_acts, id):
        boards = np.repeat(board.reshape((1,)+board.shape),num_acts,0)
        cnt = 0
        try:
            for new_board in boards:
                leaf = self.tree_policy(root, new_board)
                if self.search_over != 0: 
                    break
                if leaf.won != game_utils.Termination.going:
                    self.backup(leaf, 0.0 if leaf.won == game_utils.Termination.tie else leaf.value)
                cnt += 1
                # print(cnt)
            self.search_over += 1
        except SearchOver as e:
            self.search_over += 1
            self.absolute_release()
        
    def absolute_release(self, lock=None):
        if lock is None:
            self.absolute_release(self.eval_lock)
            for lock in self.locks:
                self.absolute_release(lock)
            return None
        try:
            lock.release()
        except:
            return None

    def simulate(self, board):
        enemy_move = self.enemy_move
        if self.last_best is None or self.inherit:
            root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        else:
            root = self.last_best.search_child_move(enemy_move)
            if root is None: root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        num_acts = self.max_acts / self.n_threads
        threads = [threading.Thread(target=self.simulate_thread_fun, args=(root, board, num_acts, i)) 
                   for i in range(self.n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # input('joined')
        self.absolute_release()
        return root

    def simulate_single_thread(self, board):
        enemy_move = self.enemy_move
        if self.last_best is None or self.inherit:
            root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        else:
            root = self.last_best.search_child_move(enemy_move)
            if root is None: root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self, self.fix_p)
        boards = np.repeat(board.reshape((1,)+board.shape),self.max_acts,0)
        for new_board in boards:
            leaf = self.tree_policy(root, new_board)
            if leaf.won != game_utils.Termination.going:
                self.backup(leaf, 0.0 if leaf.won == game_utils.Termination.tie else leaf.value)
            # No default policy
        return root

    def probs_board(self):
        return [self.root.probs(self.root_sim_board), self.root.create_eval_board(self.root_sim_board)]

    def select_action(self, board):
        self.search_over = 0
        self.root = self.simulate(board.copy())
        best, _, _ = self.root.select(board.copy())
        self.root_sim_board = board.copy()
        self.last_best = best
        return best.move

    def eval_state_lock(self, board):
        """
        return a lock and a list [value, prob]
        """
        # print('eval_lock.acquire')
        self.eval_lock.acquire()
        self.eval_queues.append(board)
        idx = len(self.eval_queues) - 1
        if idx == 0 and self.search_over == 0:
            self.eval_event.clear()
        elif self.search_over != 0:
            self.eval_event.set()
        self.eval_results[idx] = []
        if idx + 1 == self.n_threads or self.search_over != 0:
            value, policy = self.zeroNN.predict(np.array(self.eval_queues))
            self.eval_queues = []
            policy = policy.reshape([-1] + list(board.shape[:-1]))
            for i in range(idx+1):
                self.eval_results[i].append(value[i][0])
                self.eval_results[i].append(policy[i])
            self.eval_event.set()
        self.eval_lock.release()
        return self.eval_results[idx]

    def eval_state_single(self, board):
        """
        Input board is a 4*board_rows*board_rows matrix
        Use NN to evaluate the current game state, return a double-element list [value, policy]
            value is a scalar while policy is a board_rows*board_cols matrix
        """
        self.eval_lock.acquire()
        value, policy = self.zeroNN.predict(board.reshape([1]+list(board.shape)))
        self.eval_lock.release()
        return value[0][0], policy.reshape([-1] + list(board.shape[:-1]))[0]

