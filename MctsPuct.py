# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: MCTS of PUCT algorithm's implementation. Difference between UCT and PUCT is that
        PUCT uses a P as prior when selecting a child node.
"""
import numpy as np
from enum import IntEnum
import game_utils
from ZeroNN import ZeroNN
#from CNN import CNN
import time
import threading
from queue import Queue
import random

# sing_evals = 0
# multi_evals = 0


class Grid(IntEnum):
    GRID_EMP = 0
    GRID_ENY = -1
    GRID_SEL = -GRID_ENY


class SearchOver(Exception):
    def __init__(self):
        pass


class Node:
    def __init__(self, parent=None, move=None, role=None, sim_board=None, mcts=None, p=0.0):
        self.visits = 0

        # the value is used for parent's selection, is the win prob after this move
        # the parent node select according to this value
        # the value is correponsing to role
        self.value = 0.0
        # the role is the executor of the move
        self.role = role
        self.mcts = mcts
        self.parent = parent
        self.move = move
        self.move_sim_board(sim_board)
        self.won = self.mcts.check_over(sim_board, move) if move is not None else game_utils.Termination.going
        if self.won == game_utils.Termination.won:
            parent.children = [self]
        self.children = []
        self.avails = self.init_avails(sim_board) #empty grids
        self.unvisited = self.avails.copy()
        self.p = p 
        self.lock = threading.Lock()
        self.last_child = None

    def value_visit_plus(self, value, visits):
        with self.lock:
            self.value += value
            self.visits += visits
        
    def search_child_move(self, move):
        for child in self.children:
            if child.move[0] == move[0] and child.move[1] == move[1]:
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
        return sim_board

    def select(self, sim_board, noexp=False):
        # return whether expanded
        # According to PUCT algorithm, selection for simulation is NON-probabilistic! 
        with self.lock:
            if self.won != game_utils.Termination.going:
                return self, False, sim_board
            elif len(self.children) == 0:
                if noexp:
                    return self, True, sim_board
                # print("expand")
                self.expand(sim_board)
                return self, True, sim_board
        if True or self.last_child is None or\
                self.visits < 25 or 0 == self.visits % self.mcts.max_update:
                # self.mcts.max_update > (random.random() ** 2) * self.visits:
            udeno = np.sqrt(self.visits)
            if self.parent is None and self.mcts.noise != 0:
                # add noise for children of the root
                noises = np.random.dirichlet([0.03 for i in range(len(self.children))]) 
                noises = list(noises)
                best = max(self.children, key=lambda child: child.puct(udeno, noises))
            else:
                best = max(self.children, key=lambda child: child.puct(udeno))
            self.last_child = best
        else:
            best = self.last_child
        best.move_sim_board(sim_board)
        return best, False, sim_board

    def puct(self, udeno, noise=None):
        noise = 0 if noise is None else noise.pop()
        # Q + U
        return self.value / (self.visits+0.01) +\
                self.mcts.c * (self.p * (1 - self.mcts.noise) + self.mcts.noise * noise) *\
                    udeno / (self.visits+1)

    def play(self, sim_board):
        try:
            probs = self.mcts.cool_probs(
                np.array([child.visits for child in self.children], dtype=np.float64))
            # for child in self.children:
            selection = np.random.choice(range(probs.size),p=probs,size=[1])
        except:
            print("ERROR play - probs=\n", probs, sim_board, self.children)
            return self.children[0]
        return self.children[selection[0]]

    def create_eval_board(self, sim_board):
        """
        Four channels:
            1. pieces of previous player
            2. pieces of the current player
            3. the node's movement(last movement)
            4. the current player
        The board can be reversed in terms of the opponent's view
        """
        eval_board = np.zeros(shape=[sim_board.shape[0],sim_board.shape[1], 4], dtype=np.int8)

        # pieces of  the previous player
        eval_board[:,:,0][sim_board==self.role] = 1

        # pieces of  the current player
        eval_board[:,:,1][sim_board==-self.role] = 1
        if self.move is not None:
            eval_board[:,:,2][self.move[0]][self.move[1]] = 1

        # at this node, the self.role has moved, so the current player to move is the -self.role
        # if it is the first player to move next, put 1
        # the resulted evaluated value is the win rate of the current player to move
        eval_board[:,:,3] = int((self.role != Grid.GRID_SEL and self.mcts.is_first) or \
                                (self.role == Grid.GRID_SEL and not self.mcts.is_first))
        return eval_board

    def print_eval_board(self, sim_board):
        eval_board = self.create_eval_board(sim_board)
        return [eval_board[:,:,i] for i in range(4)]

    def children_values(self, sim_board):
        values = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            values[child.move[0]][child.move[1]] = child.value / (child.visits + 0.01)
        return values

    def children_visits(self, sim_board):
        visits = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            visits[child.move[0]][child.move[1]] = child.visits 
        return visits

    def probs(self, sim_board, need_cool=True):
        probs = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            # print(child.visits, id(child.parent))
            probs[child.move[0]][child.move[1]] += child.visits
        if not need_cool:
            return probs / self.visits
        # act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        # print('self.visits',self.visits, id(self), len(self.children))
        return self.mcts.cool_probs(probs)

    def default_policy(self, board):
        if not self.mcts.usedef:
            return 0.0
        bd = board.copy()
        avails = self.avails
        np.random.shuffle(avails)
        role = -self.role
        for avail in avails:
            bd[avail[0]][avail[1]] = role
            termination = self.mcts.check_over(bd, avail)
            if termination != game_utils.Termination.going:
                ret = (role == self.role) * 2 - 1 if game_utils.Termination.won == termination else 0.0
                return ret
            role = -role
        return 0.0

    def expand(self, sim_board):
        """
        We expand all children
        """
        # print('expand')
        # input()
        if self.mcts.zeroNN is None:
            # use default policy if no zeroNN is provided
            self.value = self.default_policy(sim_board)
            probs = self.mcts.defprobs
        else:
            eval_board = self.create_eval_board(sim_board)
            rets = self.mcts.eval_state_multi(eval_board)
            # avoid deadlock or long waiting:
            # if the evaluation queue cannot be full for a long time, just eval a single game board
            # the logic here should be carefully dealt with:
            #   if there is something unexpected with multi-eval, turn to single-eval
            if not self.mcts.wait_for_full():
                self.value, probs = self.mcts.eval_state_single(eval_board)
            else:
                if len(rets) == 0:
                    self.value, probs = self.mcts.eval_state_single(eval_board)
                else:
                    self.value, probs = rets
        for r,c in self.avails:
            self.children.append(Node(self, (r,c), Grid(-self.role), sim_board.copy(), self.mcts, probs[r][c]))


class MctsPuct:
    def __init__(self, board_rows_, board_cols_, n_target_=4, max_t_=5,
                 max_acts_=1000, c=2.5, inherit=True, zeroNN=None, n_threads=4,
                 multi_wait_time_s=0.030, const_temp=0, noise=0.2, temp2zero_moves=0xfffff,
                 resign_val=0.65, split=2, usedef=True, max_update=0xffff):
        """
        @board_rows_: number of rows of the game board
        @board_cols_: number of cols of the game board, recommended to be the same as board_rows_ in benefit of data augmentation
        @n_target_: number of stones in a line as win condition
        @const_temp: the temperature constant, used for prob's calculation. Set it negative 
            if you want a infinite temperature
        @c: the constant c_puct in PUCT algorithm. On a 3*3 board, it's recomended to be 0.1.
        @inherit: whether or not to inherit from parent tree. If set false, a new root would 
            be generated for every search.
        @multi_wait_time_s: the time that a thread should wait for queued evaluation 
            The greater it is, the more efficient the evaluator(neural network) will be, but 
            a thread may wait for more time.
        @n_threads: number of threads to search
        @noise: the noise applied to the prior probability P.
        @temp2zero_moves: after temp2zero_moves moves, const_temp would be set 0
        @resign_val: once enemy's win rate is evluated more than resign_val, then resign
        @max_update: the smaller it is, the more frequently the best child will be updated 
            and, as a result, more time the computation will take.
        """
        self.n_target = n_target_
        self.max_t = max_t_
        self.max_acts = max_acts_
        self.board_cols = board_cols_
        self.board_rows = board_rows_
        self.c = c
        self.last_best = None
        self.inherit = inherit
        self.zeroNN = zeroNN
        self.max_update = max_update
        self.enemy_move = None
        self.init_syncs(n_threads)
        self.multi_wait_time_s = multi_wait_time_s
        self.const_temp = const_temp
        self.split = split
        self.split_chance = 0.1 * split
        self.usedef = usedef
        self.temp_reciprocal = 1.0 / const_temp if const_temp != 0 else None
        if self.temp_reciprocal  is not None:
            self.temp_reciprocal = np.clip(self.temp_reciprocal,1e-3, 1e3)

        # virtual losses
        self.virtual_value_plus = 0
        self.virtual_visits_plus = 0.5
        self.noise = noise
        self.temp2zero_moves = temp2zero_moves
        self.moves = 0
        self.resign_val = resign_val
        self.is_first = None

    def wait_for_full(self):
        return self.eval_event.wait(self.multi_wait_time_s)

    def cool_probs(self, probs):
        """
        @probs should be a array of visits
        """
        if self.const_temp == 0:
            ret = np.zeros(probs.shape)
            m = np.argmax(probs)
            if len(ret.shape) == 1:
                ret[m] = 1.0
                return ret
            ret[m//self.board_cols][m%self.board_cols] = 1.0
            return ret
        elif self.const_temp == 1:
            return probs / np.sum(probs)
        # print("self.temp_reciprocal=", self.temp_reciprocal, type(self.temp_reciprocal))
        probs **= self.temp_reciprocal
        return probs / np.sum(probs)

    def cool_prob(self, prob):
        if self.const_temp == 0:
            return 1
        elif self.const_temp == 1:
            return prob
        return prob ** self.temp_reciprocal

    def init_syncs(self, n_threads):
        self.eval_queues = []
        self.eval_results = [[] for _ in range(n_threads)]
        self.n_threads = n_threads
        # self.threads = n_threads
        # self.locks = [threading.Lock() for _ in range(n_threads)]
        self.eval_lock = threading.Lock()
        self.eval_event = threading.Event()
        self.search_over = 0

    def from_another_mcts(self, other):
        self.max_update = other.max_update
        self.max_t = other.max_t
        self.max_acts = other.max_acts
        self.c = other.c
        self.inherit = other.inherit
        self.usedef = other.usedef
        self.zeroNN = other.zeroNN
        self.noise = other.noise
        self.const_temp = other.const_temp
        self.temp_reciprocal = other.temp_reciprocal
        self.temp2zero_moves = other.temp2zero_moves
        self.resign_val = other.resign_val
        self.split = other.split
        self.split_chance = other.split_chance

    def copy_params(self, other):
        self.from_another_mcts(other)

    def tree_policy(self, root, sim_board):
        roots = [root]
        selects = []
        noexp = False
        # print("tree_policy")
        while len(roots) != 0:
            reach_leaf = False
            selected = roots.pop()
            while not reach_leaf and selected.won == game_utils.Termination.going:
                if len(selects) == 0:
                    selected.value_visit_plus(self.virtual_value_plus, self.virtual_visits_plus)
                selected, reach_leaf, sim_board = selected.select(sim_board, noexp)
                if len(selects) < self.split and np.random.rand() < self.split_chance:
                    roots.append(selected)
            selects.append(selected)
            if reach_leaf:
                noexp = True
        return selects

    def default_policy(self, node):
        """
        MctsPuct does not use default_policy according to alphaGo Zero
        """
        bd = node.sim_board
        avails = node.avails
        np.random.shuffle(avails)
        role = Grid(-node.role)
        for avail in avails:
            bd[avail[0]][avail[1]] = role
            termination = self.check_over(bd, avail)
            if termination != game_utils.Termination.going:
                return role if game_utils.Termination.won == termination else None
            role = Grid(-role)
        return None

    def check_over(self, bd, pos):
        return game_utils.Game.check_over_full(bd, pos, self.n_target)

    def backup(self, nodes):
        """
        If win_role is GRID_SEL, win_rate should have the same sign as GRID_SEL
        If win_role is GRID_ENY, win_rate should have the same sign as GRID_ENY
        The trick can simplify the backup process
        """
        virtual_value_plus = self.virtual_value_plus
        virtual_visits_plus = self.virtual_visits_plus
        for node in nodes:
            win_rate = node.value * node.role if node.won == game_utils.Termination.going \
                else float(0 if node.won == game_utils.Termination.tie else node.role)
            # if this is a newly expanded node, 
            #   do not back up value since its value has been updated on expansion
            #   and do not apply virtual loss to visits since this is its first visit
            if abs(node.visits - self.virtual_visits_plus) < 1e-6 and virtual_visits_plus > 0:
                node.value_visit_plus(0, 1-virtual_visits_plus)
                node = node.parent
            while node != None:
                node.value_visit_plus(node.role * win_rate-virtual_value_plus, 1-virtual_visits_plus)
                node = node.parent
            virtual_value_plus = 0
            virtual_visits_plus = 0

    def simulate_thread_fun(self, root, board, num_acts, id):
        boards = np.repeat(board.reshape((1,)+board.shape),num_acts,0)
        for new_board in boards:
            leaves = self.tree_policy(root, new_board)
            self.backup(leaves)
        self.search_over += 1

    def simulate(self, board):
        enemy_move = self.enemy_move
        if self.last_best is None or not self.inherit:
            root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self)
        else:
            # print("enemy_move=",enemy_move)
            root = self.last_best.search_child_move(enemy_move)
            # print("root=",root)
            if root is None: root = Node(None, enemy_move, Grid.GRID_ENY, board.copy(), self)
            # else: print("GOT root")
            # input()
        root.parent = None
        num_acts = self.max_acts / self.n_threads
        self.tmp = root
        threads = [threading.Thread(target=self.simulate_thread_fun, args=(root, board, num_acts, i)) 
                   for i in range(self.n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return root

    def probs_board(self):
        return [self.root.probs(self.root_sim_board, need_cool=False), 
                self.root.create_eval_board(self.root_sim_board)]

    def select_action(self, board):
        """
        return None for resign
        """
        t = time.time()
        if self.is_first is None:
            self.is_first = (np.sum(board) == 0)
            self.defprobs = np.zeros((self.board_rows, self.board_cols)) +\
                1.0/ self.board_cols / self.board_rows
        self.eval_queues = []
        self.search_over = 0
        self.root = self.simulate(board.copy())
        best = self.root.play(board.copy())
        self.root_sim_board = board.copy()
        self.last_best = best
        self.moves += 1
        if self.moves > self.temp2zero_moves:
            self.const_temp = 0
        '''
        if self.zeroNN is not None:
            value, policy = self.eval_state_single(self.root.create_eval_board(board.copy()))
        else:
            value = 0.0
            policy = 0.0
        np.set_printoptions(4, suppress=True)
        print('self.root.value:',self.root.value /self.root.visits )
        print('value:',value)
        print('policy:\n',policy)
        print('probs:\n',self.root.probs(board))
        print("root.children_values")
        print(self.root.children_values(board))
        print("root.children_visits")
        print(self.root.children_visits(board))
        print("time=",time.time()-t)
        # print(self.root.print_eval_board(self.root_sim_board))
        # m = board.copy()
        # best.move_sim_board(m)
        # print("best")
        
        # print(self.root.role)
        # print(self.root.print_eval_board(self.root_sim_board))
        # print(self.last_best.role)
        # print(self.last_best.print_eval_board(best.move_sim_board(board.copy())))
        # print('selection used',time.time() - t, multi_evals, sing_evals)
        input()
        '''

        if self.root.value /self.root.visits >= self.resign_val:
            return None
        
        # input()
        # print('selection used',time.time()-t ,'  single:',single_cnt,'   multi:',multi_cnt)
        return best.move

    def eval_state_multi(self, board):
        """
        return a lock and a list [value, prob]
        """
        # print('eval_lock.acquire')
        self.eval_lock.acquire()
        # global multi_evals
        # multi_evals += 1
        self.eval_queues.append(board)
        idx = len(self.eval_queues) - 1

        # if there is a thread that finished searching, do not wait the eval_queue to be full
        if idx == 0 and self.search_over == 0:
            self.eval_event.clear()
        elif self.search_over != 0:
            self.eval_event.set()
        self.eval_results[idx] = []
        lres = len(self.eval_results)
        if idx + 1 == lres or self.search_over != 0:
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
        Input board is a board_rows*board_cols*4 matrix
        Use NN to evaluate the current game state, return a double-element list [value, policy]
            value is a scalar while policy is a board_rows*board_cols matrix
        """
        value, policy = self.zeroNN.predict(board.reshape([1]+list(board.shape)))
        return value[0][0], policy.reshape([-1] + list(board.shape[:-1]))[0]
