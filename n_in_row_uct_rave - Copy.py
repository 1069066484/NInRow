
"""
author: arrti
github: https://github.com/arrti
blog:   http://www.cnblogs.com/xmwd
""" 

import copy
import time
from random import choice, shuffle
from math import log, sqrt
from MCTS_raw import MCTS


class Board(object):
    """
    board for game
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # board states, key:move, value: player as piece type
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win

    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)

        self.availables = list(range(self.width * self.height)) # available moves

        self.states = {} # key:move as location on the board, value:player as pieces type

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def update(self, player, move):
        self.states[move] = player
        self.availables.remove(move)

    def current_state(self):
        return tuple((m, self.states[m]) for m in sorted(self.states)) # for hash


class Human(object):
    """
    human player
    """

    def __init__(self, board, player):
        self.board = board
        self.player = player

    def get_action(self):
        try:
            location = [int(n, 10) for n in input("Your move: ").split(",")]
            move = self.board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in self.board.availables:
            print("invalid move")
            move = self.get_action()
        return move

    def __str__(self):
        return "Human"


class Game(object):
    """
    game server
    """

    def __init__(self, board, **kwargs):
        self.board = board
        self.player = [1, 2] # player1 and player2
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.time = float(kwargs.get('time', 5))
        self.max_actions = int(kwargs.get('max_actions', 1000))

    def start(self):
        p1, p2 = self.init_player()
        self.board.init_board()

        ai = MCTS(self.board, [p1, p2], self.n_in_row, self.time, self.max_actions)
        human = Human(self.board, p2)
        players = {}
        players[p1] = ai
        players[p2] = human
        turn = [p1, p2]
        shuffle(turn)
        self.graphic(self.board, human, ai)
        while(1):
            p = turn.pop(0)
            turn.append(p)
            player_in_turn = players[p]
            move = player_in_turn.get_action()
            self.board.update(p, move)
            self.graphic(self.board, human, ai)
            end, winner = self.game_end(ai)
            if end:
                if winner != -1:
                    print("Game end. Winner is", players[winner])
                break

    def init_player(self):
        plist = list(range(len(self.player)))
        index1 = choice(plist)
        plist.remove(index1)
        index2 = choice(plist)

        return self.player[index1], self.player[index2]

    def game_end(self, ai):
        win, winner = ai.has_a_winner(self.board)
        if win:
            return True, winner
        elif not len(self.board.availables):
            print("Game end. Tie")
            return True, -1
        return False, -1

    def graphic(self, board, human, ai):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height

        print("Human Player", human.player, "with X".rjust(3))
        print("AI    Player", ai.player, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == human.player:
                    print('X'.center(8), end='')
                elif p == ai.player:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')


def run():
    n = 4
    try:
        board = Board(width=5, height=5, n_in_row=n)
        game = Game(board, n_in_row=n, time=5) # more time better
        game.start()
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    run()