# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the NinRowAI-AlphaRow: MCTS of UCT algorithm's implementation.
"""
import numpy as np
from enum import IntEnum


n_target = None
max_t = None
max_acts = None
board_cols = None
board_rows = None
GRID_EMP = 0
GRID_ENY = -1
GRID_SEL = -GRID_ENY
c = 1
curr_root = None


class Termination(IntEnum):
    going = 100
    won = 101
    tie = 102


def init(board_cols_, board_rows_, n_target_=5, max_t_=5, max_acts_=1000):
    global board_cols, board_rows, n_target, max_t, max_acts
    n_target = n_target_
    max_t = max_t_
    max_acts = max_acts_
    board_cols = board_cols_
    board_rows = board_rows_


def set_max_act(max_acts_):
    global max_acts
    max_acts = max_acts_


class Node:
    def __init__(self, parent=None, move=None, role=None, sim_board=None):
        self.visits = 0
        self.value = 0.0
        self.role = role
        self.parent = parent
        self.move = move
        self.move_sim_board(sim_board)
        self.won = check_over(sim_board, move) if move is not None else Termination.going
        self.childrens = []
        self.avails = self.init_avails() #empty grids
        self.unvisited = self.avails.copy()

    def init_avails(self):
        avails = []
        if self.parent is not None:
            avails = self.parent.avails.copy()
            avails.remove(self.move)
        else:
            for r in range(board_rows):
                for c in range(board_cols):
                    if self.sim_board[r][c] == GRID_EMP:
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
        elif self.won != Termination.going:
            return self, False
        best = max(self.childrens, key=lambda child: \
                    child.value/(child.visits+0.01)+c*np.sqrt( np.log(self.visits+1.1)/ (child.visits+0.01)))
        best.move_sim_board(self.sim_board)
        return best, False

    def expand(self):
        selected = self.unvisited[-1]
        self.unvisited.pop()
        self.childrens.append(Node(self, selected, -self.role, self.sim_board))
        return self.childrens[-1]


def tree_policy(root):
    expanded = False
    selected = root
    while not expanded:
        if selected.won != Termination.going:
            return selected
        selected, expanded = selected.select()
        # print(selected.role)
        # print(selected.role, selected.move)
    return selected


def default_policy(node):
    bd = node.sim_board
    avails = node.avails
    np.random.shuffle(avails)
    role = -node.role
    for avail in avails:
        bd[avail[0]][avail[1]] = role
        termination = check_over(bd, avail)
        if termination != Termination.going:
            # print(bd, role, n_target)
            return role if Termination.won == termination else None
        role = -role
    return None


over_funs = [
    lambda i,p:(p[0]+i,p[i]+i),
    lambda i,p:(p[0]-i,p[i]+i),
    lambda i,p:(p[0],p[i]+i),
    lambda i,p:(p[0]+i,p[i])
    ]


def check_over(bd, pos):
    return check_over_full(bd, pos, n_target)


def check_over_full(board, pos, targets):
    """
    return Termination
    """
    def is_pos_legal(pos):
        return board.shape[0] > pos[0] >= 0 and board.shape[1] > pos[1] >= 0
    bd = board
    for f in over_funs:
        role = bd[pos[0]][pos[1]]
        score = 0
        pos_t = pos
        while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
            score += 1
            pos_t = f(1,pos_t)
        pos_t = f(-1,pos)
        while is_pos_legal(pos_t) and bd[pos_t[0]][pos_t[1]] == role:
            score += 1
            pos_t = f(-1,pos_t)
        if score >= targets:
            return Termination.won
    return Termination.going if np.sum(bd != GRID_EMP) != bd.size else Termination.tie


def backup(node, win_role):
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


def simulate(board):
    global curr_root
    if curr_root is None or True:
        root = Node(None, None, -GRID_SEL, board.copy())
    else:
        for child in curr_root.childrens:
            if board[child.move[0]][child.move[1]] == GRID_ENY:
                root = child
                break
    win_cnt = 0
    t_sim = 0
    for board in np.repeat(board.reshape((1,)+board.shape),max_acts,0):
        # print('\n')
        if t_sim % 2000 == 0 and t_sim != 0:
            print('win rate:', win_cnt/t_sim)
        t_sim += 1
        root.sim_board = board
        leaf = tree_policy(root)
        if leaf.won != Termination.going:
            win_cnt += (leaf.role == GRID_SEL and leaf.won == Termination.won)
            backup(leaf, None if leaf.won == Termination.tie else leaf.role)
            # print(leaf.role)
            continue
        win_role = default_policy(leaf)
        if win_role != 0:
            backup(leaf, win_role)
            win_cnt += (win_role == GRID_SEL)
            # print(win_role)
    # print("win_cnt=",win_cnt)
    curr_root = root
    return root



def select_action(board):
    root = simulate(board)
    best, _ = root.select()
    return best.move



player = None
player_op = None
itf_board = None
def interface_get_action():
    bd = interface_board(itf_board)
    print(bd)
    move = select_action(bd)
    return move[0] * board_cols + move[1]

# MCTS(self.board, [p1, p2], self.n_in_row, self.time, self.max_actions)

def interface_init(board, players, n_in_rows, time, max_actions):
    global player, player_op, itf_board
    print(board.width, board.height)
    init(board.width, board.height, n_in_rows, time, max_actions)
    print(board_rows, board_cols)
    itf_board = board
    player = players[0]
    player_op = players[1]


def interface_board(board):
    bd = np.zeros([board_rows, board_cols], dtype=np.int8)
    for k,v in board.states.items():
        bd[k // board_cols][k % board_cols] = GRID_SEL if v == player else GRID_ENY
    return bd


def interface_has_a_winner(move):
    winner = check_over(interface_board(itf_board), (move // board_cols, move % board_cols))
    if winner == GRID_SEL:
        return True, player
    elif winner == GRID_ENY:
        return True, player_op
    else:
        return False, -1



