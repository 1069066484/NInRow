# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI: MCTS of PUCT algorithm's implementation. Difference between UCT and PUCT is that
        PUCT uses a P as prior when selecting a child node.
"""
import numpy as np
import time
import threading
import random
import copy
from games.Termination import *
from utils.data import *


CHECK_DETAILS = False
EVAL_HIST = 1
T_SELECT = [0,0]


class Node:
    def __init__(self, parent, move, role, sim_board, mcts, p=0.0, stones=0):
        self.visits = 0
        # the role is the executor of the move
        self.role = role
        self.mcts = mcts
        self.parent = parent
        self.move = move
        self.stones = stones
        # the value is used for parent's selection, is the win prob after this move
        # the parent node select according to this value
        # the value is correponsing to role
        won_val_self = self.won_val(role, sim_board)
        self.won = won_val_self[0]
        self.value = max(won_val_self[1], self.won_val(-role, sim_board)[1] * 0.5)
        # print(self.stones, self.value)
        self.move_sim_board(sim_board)
        self.children = []
        # t = time.clock()
        self.avails = self.init_avails(sim_board) #empty grids
        self.p = p 
        self.lock = threading.Lock()
        self.move_sim_board(sim_board, True)
        self.last_best = None
        # T_SELECT[0] += time.clock() - t

    def prune(self):
        # return
        if self.mcts.do_prune and len(self.children) > 1:
            mv = max(self.children, key=lambda child: child.value).value
            if mv >= 0.9999:
                # print("mv > 0.9999")
                self.children = [child for child in self.children if child.value >= 0.9]
                # self.pr = 'Almost Win'
            elif mv >= DECIDE_VAL:
                # print("mv > DECIDE_VAL")
                self.children = [child for child in self.children if (child.value >= DECIDE_VAL * 0.4)]
                # self.pr = 'Advantage'
            elif self.mcts.zeroNN is not None and len(self.children) >= 4:
                # print("policy prune")
                pmax = max(self.children, key=lambda child: child.p).p
                if pmax > 0.1:
                    pmax *= 0.05
                    self.children = [child for child in self.children if (child.p >= pmax)]

    def won_val(self, role, sim_board):
        if self.move is None:
            return [Termination.going, 0]
        sim_board[self.move[0]][self.move[1]] = role
        return self.mcts.check_over(sim_board, self.move, True, self.stones)


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
        if self.parent is not None:
            avails = self.parent.avails.copy()
            avails.remove(self.move)
        else:
            avails = []
            for r in range(sim_board.shape[0]):
                for c in range(sim_board.shape[1]):
                    if sim_board[r][c] == GRID_EMP_INT:
                        avails.append((r,c))
        return avails

    def move_sim_board(self, sim_board, rollback=False):
        if self.move is not None:
            sim_board[self.move[0]][self.move[1]] = GRID_EMP_INT if rollback else self.role

    def select(self, sim_board, noexp=False):
        '''
        return [the leaf node, whether expanded]
        The leaf node has just expanded.
        According to PUCT algorithm, selection for simulation is NON-probabilistic! 
        '''
        with self.lock:
            if self.won != Termination.going:
                return self, False, sim_board
            elif len(self.children) == 0:
                if noexp and self.parent is not None:
                    return self, True, sim_board
                # print("expand")
                self.expand(sim_board)
                return self, True, sim_board
        best = self.children_qu(self.parent is None and self.mcts.noise != 0)
        best.move_sim_board(sim_board)
        return best, False, sim_board

    def children_qu(self, need_noise=False):
        noise = np.random.dirichlet([0.03 for i in range(len(self.children))]) \
            if need_noise else None
        best = self.children[0]
        udeno = np.sqrt(self.visits)
        for i in range(len(self.children)):
            if self.children[i].puct(udeno, noise[i] if need_noise else None) >= best.qu:
                best = self.children[i]
        return best

    def puct(self, udeno, noise=None):
        # Q + U
        if noise is None:
            self.qu = self.value / (self.visits+0.01) +  self.mcts.c * self.p * udeno / (self.visits+1)
        else:
            self.qu = self.value / (self.visits+0.01) +\
                    self.mcts.c * (self.p * (1 - self.mcts.noise) + self.mcts.noise * noise) *\
                        udeno / (self.visits+1)
        return self.qu

    def play(self):
        probs = None
        try:
            # If two nodes have the same number of visits, select the one that has larger value
            if self.mcts.const_temp == 0:
                return max(self.children, key=lambda child: child.visits + child.value / (child.visits + 1) * 0.001 + 0.01)
            probs = self.mcts.cool_probs(
                np.array([child.visits + child.value / (child.visits + 1) * 0.001 + 0.01 for child in self.children], dtype=np.float64))
            # for child in self.children:
            selection = np.random.choice(range(probs.size),p=probs,size=[1])
        except:
            print("ERROR play - probs=\n", probs, self.children, self.avails)
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
        eval_board = np.zeros(shape=[sim_board.shape[0],sim_board.shape[1], 3 + EVAL_HIST], dtype=np.bool)

        # stones of the previous player
        eval_board[:,:,0][sim_board==self.role] = 1

        # stones of the current player
        eval_board[:,:,1][sim_board==-self.role] = 1
        if self.move is not None:
            eval_board[:,:,2][self.move[0]][self.move[1]] = 1
        if EVAL_HIST == 2 and self.parent is not None and self.parent.move is not None:
            eval_board[:,:,3][self.parent.move[0]][self.parent.move[1]] = 1
        # at this node, the self.role has moved, so the current player to move is the -self.role
        # if it is the first player to move next, put 1
        # the resulted evaluated value is the win rate of the current player to move
        eval_board[:,:,-1] = int((self.role != GRID_SEL_INT and self.mcts.is_first) or \
                                (self.role == GRID_SEL_INT and not self.mcts.is_first))
        return eval_board

    def print_eval_board(self, sim_board):
        eval_board = self.create_eval_board(sim_board)
        return [eval_board[:,:,i] for i in range(4)]

    def children_values(self, sim_board):
        values = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            values[child.move[0]][child.move[1]] = child.value / child.visits if child.visits >= 1 else 0
        return values

    def children_visits(self, sim_board):
        visits = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            visits[child.move[0]][child.move[1]] = child.visits 
        return visits

    def probs(self, sim_board, need_cool=True):
        probs = np.zeros(sim_board.shape, dtype=np.float64)
        for child in self.children:
            probs[child.move[0]][child.move[1]] += child.visits + child.value / (child.visits + 1) * 0.001 + 0.01
        if not need_cool:
            return probs / self.visits
        return self.mcts.cool_probs(probs)

    def default_policy(self, board):
        if self.mcts.usedef is None:
            return 0.0
        bd = board
        avails = [self.avails[i] for i in \
                  np.random.permutation(len(self.avails))[:min(self.mcts.usedef[0], len(self.avails))]]
        role = -self.role
        mult = (1.0 / len(self.avails)) ** 0.5
        ret = 0.0
        for avail in avails:
            bd[avail[0]][avail[1]] = role
            termination, val = self.mcts.check_over(bd, avail, True)
            if abs(val) > DECIDE_VAL:
                ret = (val if (role == self.role) else -val)
                break
            if termination != Termination.going:
                ret = (role == self.role) * 2 - 1 if Termination.won == termination else 0.0
                break
            role = -role
        for avail in avails:
            bd[avail[0]][avail[1]] = GRID_EMP_INT
        return ret * mult

    def expand(self, sim_board):
        """
        We expand all children
        """
        if self.mcts.zeroNN is None:
            # use default policy if no zeroNN is provided
            if self.mcts.usedef is not None:
                self.value = (self.value + \
                              np.mean([self.default_policy(sim_board) for _ in range(self.mcts.usedef[1])])) / 2
            probs = self.mcts.defprobs
        else:
            # self.sim_board = sim_board.copy()
            eval_board = self.create_eval_board(sim_board)
            rets = self.mcts.eval_state_multi(eval_board)
            # avoid deadlock or long waiting:
            # if the evaluation queue cannot be full for a long time, just eval a single game board
            # the logic here should be carefully dealt with:
            #   if there is something unexpected with multi-eval, turn to single-eval
            if not self.mcts.wait_for_full():
                value, probs = self.mcts.eval_state_single(eval_board)
            else:
                if len(rets) == 0:
                    value, probs = self.mcts.eval_state_single(eval_board)
                else:
                    value, probs = rets
            self.value = self.value * self.mcts.hand_val +  value * (1.0 - self.mcts.hand_val)
        stones = self.stones + 1
        self.children = [Node(self, (r,c), Grid(-self.role), sim_board, self.mcts, probs[r][c], stones) 
                         for r,c in self.avails]
        self.prune()
        random.shuffle(self.children)


class MctsPuct:
    def __init__(self, board_rows_, board_cols_, n_target_=4, max_acts_=1000, c=10, inherit=True, zeroNN=None, n_threads=4,
                 multi_wait_time_s=0.030, const_temp=0, noise=None, temp2zero_moves=0xfffff,
                 resign_val=0xffff, usedef=None, do_prune=True, hand_val=0.1, d_time=0):
        """
        @board_rows_: number of rows of the game board
        @board_cols_: number of cols of the game board, recommended to be the same as board_rows_ in benefit of data augmentation
        @n_target_: number of stones in a line as win condition
        @max_acts: simulations performed to play an action
        @c: the constant c_puct in PUCT algorithm. On a 3*3 board, it's recomended to be 0.1. The larger the board is,
            the greater value c is recommended to be.
        @inherit: whether or not to inherit from parent tree. If set false, a new root would 
            be generated for every search. Recommended to be True.
        @zeroNN: zeroNN used to evaluate value and policy.
        @n_threads: number of threads to search
        @multi_wait_time_s: the time that a thread should wait for queued evaluation 
            The greater it is, the more efficient the evaluator(neural network) will be, but 
            a thread may wait for more time.
        @const_temp: the temperature constant, used for prob's calculation. Set it negative 
            if you want a infinite temperature
        @noise: the noise applied to the prior probability P. Attention: by setting noise non-zero, strength of AI will degrade
            fast, so do so only in self-play stage.
        @temp2zero_moves: after temp2zero_moves moves, const_temp would be set 0
        @resign_val: once enemy's win rate is evluated more than resign_val, then resign
        @usedef: set None not to use default policy, or False no to use handcrafted estimated value
        @do_prune: whether or not to prune
        @hand_val: proportion of handcrafted value used along with network-evaluated value.
        @d_time: time counter, used for debugging
        """
        self.n_target = n_target_
        self.max_acts = max_acts_
        self.board_cols = board_cols_
        self.board_rows = board_rows_
        self.c = c
        self.last_best = None
        self.inherit = inherit
        self.do_prune = do_prune
        self.zeroNN = zeroNN
        self.enemy_move = None
        self.init_syncs(n_threads)
        self.multi_wait_time_s = multi_wait_time_s
        self.const_temp = const_temp
        self.usedef = usedef if isinstance(usedef, list) or usedef is None else [usedef, 1] 
        self.temp_reciprocal = 1.0 / const_temp if const_temp != 0 else None
        if self.temp_reciprocal  is not None:
            self.temp_reciprocal = np.clip(self.temp_reciprocal,1e-3, 1e3)
        self.hand_val = hand_val
        # virtual losses
        self.virtual_value_plus = -0.5
        self.virtual_visits_plus = 0.3
        self.noise = noise
        self.temp2zero_moves = temp2zero_moves
        self.moves = 0
        self.resign_val = resign_val
        self.is_first = None
        self.d_time = d_time

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
        self.eval_lock = threading.Lock()
        self.eval_event = threading.Event()
        self.search_over = 0

    def from_another_mcts(self, other):
        self.max_acts = other.max_acts
        self.c = other.c
        self.inherit = other.inherit
        self.usedef = other.usedef
        self.do_prune = other.do_prune
        self.zeroNN = other.zeroNN
        self.noise = other.noise
        self.const_temp = other.const_temp
        self.temp_reciprocal = other.temp_reciprocal
        self.temp2zero_moves = other.temp2zero_moves
        self.resign_val = other.resign_val
        self.hand_val = other.hand_val

    def copy_params(self, other):
        self.from_another_mcts(other)

    def tree_policy(self, root, sim_board):
        '''
        Virtual losses are not applied to the leaf and the root
        '''
        node = root
        while True:
            retnode, reach_leaf, sim_board = node.select(sim_board)
            if reach_leaf or retnode.won != Termination.going:
                if node == retnode and node != root:
                    # a leaf does not need virtual losses since it can disrupt evaluated value
                    retnode.value_visit_plus(-self.virtual_value_plus, -self.virtual_visits_plus)
                return retnode
            node = retnode
            node.value_visit_plus(self.virtual_value_plus, self.virtual_visits_plus)

    def check_over(self, bd, pos, ret_val=False, stones=None):
        return check_over_full(bd, pos, self.n_target, ret_val, stones)

    def backup(self, node, bd):
        """
        If win_role is GRID_SEL, win_rate should have the same sign as GRID_SEL
        If win_role is GRID_ENY, win_rate should have the same sign as GRID_ENY
        The trick can simplify the backup process
        """
        win_rate = node.value * node.role
        node.value_visit_plus(0, 1)
        node = node.parent

        # this node is right the root
        if node is None:
            return
        while node.parent is not None:
            node.value_visit_plus(node.role * win_rate-self.virtual_value_plus, 1-self.virtual_visits_plus)
            node = node.parent
        # Handle the root
        node.value_visit_plus(node.role * win_rate, 1) 

    def simulate_thread_fun(self, root, board, num_acts, id):
        boards = np.repeat(board.reshape((1,)+board.shape),num_acts,0)
        for new_board in boards:
            # print(new_board.shape)
            leaf = self.tree_policy(root, new_board)
            self.backup(leaf, new_board)
        self.search_over += 1

    def simulate(self, board):
        enemy_move = self.enemy_move
        if self.last_best is None or not self.inherit:
            root = Node(None, enemy_move, GRID_ENY_INT, board, self, stones=self.moves * 2 + 1 - int(self.is_first))
        else:
            root = self.last_best.search_child_move(enemy_move)
            if root is None: 
                root = Node(None, enemy_move, GRID_ENY_INT, board, self, stones=self.moves * 2 + 1 - int(self.is_first))

        if enemy_move is not None:
            board[enemy_move[0]][enemy_move[1]] = GRID_ENY_INT
        root.parent = None
        num_acts = self.max_acts // self.n_threads
        threads = [threading.Thread(target=self.simulate_thread_fun, args=(root, board.copy(), num_acts, i)) 
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
        t = time.clock()
        if self.is_first is None:
            self.is_first = (np.sum(board) == 0)
            self.defprobs = np.zeros((self.board_rows, self.board_cols)) +\
                1.0/ self.board_cols / self.board_rows
            if self.noise is None:
                if self.zeroNN is not None:
                    self.noise = 0.01
                else:
                    self.noise = 0.1/ self.board_cols / self.board_rows
        self.eval_queues = []
        self.search_over = 0
        self.root = self.simulate(board.copy())
        best = self.root.play()
        self.root_sim_board = board.copy()
        self.last_best = best
        self.moves += 1
        if self.moves > self.temp2zero_moves:
            self.const_temp = 0
        # self.d_time += time.clock() - t
        if CHECK_DETAILS:
            if self.zeroNN is not None:
                value, policy = self.eval_state_single(self.root.create_eval_board(board.copy()))
            else:
                value = 0.0
                policy = 0.0
            np.set_printoptions(2, suppress=True)
            '''
                        self.qu = self.value / (self.visits+0.01) +\
                    self.mcts.c * (self.p * (1 - self.mcts.noise) + self.mcts.noise * noise) *\
                        udeno / (self.visits+1)
            '''
            print("T_SELECT=", T_SELECT, "  Termination T=", TERM_T)
            print("self.root.visits=", self.root.visits)
            print('self.root.value:',self.root.value /self.root.visits)
            print('best avg val=', best.value / (best.visits+0.01),
                  '  cp=', np.sqrt(self.root.visits) / (best.visits + 1) * best.p * self.c, '  c=', self.c)
            print('estimated value:',value)
            bd = board.copy()
            bd[best.move[0]][best.move[1]] = best.role
            print('handcrafted value:',self.check_over(bd, best.move, True)[1])
            print('policy:\n',policy)
            print('probs:\n',self.root.probs(board))
            print("root.children_values, children_visits", len(self.root.children))
            print(self.root.children_values(board))
            print(self.root.children_visits(board))
            print("time=",time.clock()-t)
            print("best:", len(best.children))
            print(best.play().move, '->', best.play().play().move)
            print(best.children_values(board))
            print(best.children_visits(board))
            input()

        if self.root.value >= self.resign_val * self.root.visits:
            return None
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



def _test_range():
    t = time.time()


if __name__=='__main__':
    pass