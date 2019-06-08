import copy
import time
from random import choice, shuffle
from math import log, sqrt


class MCTS(object):
    """
    AI player, UCT RAVE
    """

    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=1000):
        self.board = board
        self.play_turn = play_turn
        self.calculation_time = float(time)
        self.max_actions = max_actions
        self.n_in_row = n_in_row

        self.player = play_turn[0] # AI is first at now
        self.confident = 1.96
        self.equivalence = 1000
        self.max_depth = 1

        self.plays = {}      # key:(action, state), value:visited times
        self.wins = {}       # key:(action, state), value:win times
        self.plays_rave = {} # key:(move, state), value:visited times
        self.wins_rave = {}  # key:(move, state), value:{player: win times}

    def get_action(self):
        if len(self.board.availables) == 1:
            return self.board.availables[0]
        
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
        # while simulations < 10000: # need many simulations for a better selection
            board_copy = copy.deepcopy(self.board)  # simulation will change board's states,
            play_turn_copy = copy.deepcopy(self.play_turn) # and play turn
            self.run_simulation(board_copy, play_turn_copy)
            simulations += 1

        print("total simulations=", simulations)

        move = self.select_one_move()
        location = self.board.move_to_location(move)
        print('Maximum depth searched:', self.max_depth)

        print("AI move: %d,%d\n" % (location[0], location[1]))

        self.prune()

        return move

    def run_simulation(self, board, play_turn):
        """
        UCT RAVE main process
        """

        plays = self.plays
        wins = self.wins
        plays_rave = self.plays_rave
        wins_rave = self.wins_rave
        availables = board.availables

        player = self.get_player(play_turn)
        visited_states = set()
        winner = -1
        expand = True
        states_list = []
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # if all moves have statistics info, choose one that have max UCB value
            state = board.current_state()
            actions = [(move, player) for move in availables]
            if all(plays.get((action, state)) for action in actions):
                total = 0
                for a, s in plays:
                    if s == state:
                        total += plays.get((a, s)) # N(s)
                beta = self.equivalence/(3 * total + self.equivalence)

                value, action = max(
                    ((1 - beta) * (wins[(action, state)] / plays[(action, state)]) +
                     beta * (wins_rave[(action[0], state)][player] / plays_rave[(action[0], state)]) + 
                     sqrt(self.confident * log(total) / plays[(action, state)]), action)
                    for action in actions)   # UCT RAVE

            else:
                # a simple strategy
                # prefer to choose the nearer moves without statistics,
                # and then the farthers.
                # try ro add statistics info to all moves quickly
                # adjacents = []
                # if len(availables) > self.n_in_row:
                #     adjacents = self.adjacent_moves(board, state, player, plays)

                # if len(adjacents):
                #     action = (choice(adjacents), player)
                # else:
                #     peripherals = []
                #     for action in actions:
                #         if not plays.get((action, state)):
                #             peripherals.append(action)
                #     action = choice(peripherals)
                action = choice(actions)           
            
            move, p = action
            board.update(player, move)

            # Expand
            # add only one new child node each time
            if expand and (action, state) not in plays:
                expand = False
                plays[(action, state)] = 0
                wins[(action, state)] = 0

                if t > self.max_depth:
                    self.max_depth = t

            states_list.append((action, state)) # states in one simulation by order of visited
            # for i, (m_root, s_root) in enumerate(states_list):
            #     for (m_sub, s_sub) in states_list[i:]:
            #         if (m_sub, s_root) not in plays_rave:
            #             plays_rave[(m_sub, s_root)] = 0 
            #             wins_rave[(m_sub, s_root)] = {}              
            #             for p in self.play_turn:
            #                 wins_rave[(m_sub, s_root)][p] = 0

            # AMAF value
            # next (action, state) is child node of all previous (action, state) nodes
            for (m, pp), s in states_list:
                if (move, s) not in plays_rave:
                    plays_rave[(move, s)] = 0 
                    wins_rave[(move, s)] = {}              
                    for p in self.play_turn:
                        wins_rave[(move, s)][p] = 0

            visited_states.add((action, state))

            is_full = not len(availables)
            win, winner = self.has_a_winner(board)
            if is_full or win:
                break

            player = self.get_player(play_turn)

        # Back-propagation
        for i, ((m_root, p), s_root) in enumerate(states_list):
            action = (m_root, p)
            if (action, s_root) in plays:
                plays[(action, s_root)] += 1 # all visited moves
                if player == winner and player in action:
                    wins[(action, s_root)] += 1 # only winner's moves

            for ((m_sub, p), s_sub) in states_list[i:]:
                plays_rave[(m_sub, s_root)] += 1 # all child nodes of s_root 
                if winner in wins_rave[(m_sub, s_root)]:                
                    wins_rave[(m_sub, s_root)][winner] += 1 # each node is divided by the player 

    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p

    def select_one_move(self):
        """
        select by win percentage 
        """
        percent_wins, move = max(
            (self.wins.get(((move, self.player), self.board.current_state()), 0) /
             self.plays.get(((move, self.player), self.board.current_state()), 1),
             move)
            for move in self.board.availables)

        # display the statistics for each possible play,
        # first is MC value, second is AMAF value
        for x in sorted(
                ((100 * self.wins.get(((move, self.player), self.board.current_state()), 0) /
                        self.plays.get(((move, self.player), self.board.current_state()), 1),
                  100 * self.wins_rave.get((move, self.board.current_state()), {}).get(self.player, 0) /
                        self.plays_rave.get((move, self.board.current_state()), 1),
                  self.wins.get(((move, self.player), self.board.current_state()), 0),
                  self.plays.get(((move, self.player), self.board.current_state()), 1),
                  self.wins_rave.get((move, self.board.current_state()), {}).get(self.player, 0),
                  self.plays_rave.get((move, self.board.current_state()), 1), 
                  self.board.move_to_location(move))
                 for move in self.board.availables),
                reverse=True):
            print('{6}: {0:.2f}%--{1:.2f}% ({2} / {3})--({4} / {5})'.format(*x))

        return move

    def adjacent_moves(self, board, state, player, plays):
        """
        adjacent moves without statistics info
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height

        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # right
            if w > 0:
                adjacents.add(m - 1) # left
            if h < height - 1:
                adjacents.add(m + width) # upper
            if h > 0:
                adjacents.add(m - width) # lower
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # upper right
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # upper left
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # lower right
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # lower left

        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get(((move, player), state)):
                adjacents.remove(move)
        return adjacents

    def prune(self):
        """
        remove not selected path
        """
        length = len(self.board.states)
        keys = list(self.plays)
        for a, s in keys:
            if len(s) < length + 2:
                del self.plays[(a, s)]
                del self.wins[(a, s)]

        keys = list(self.plays_rave)
        for m, s in keys:
            if len(s) < length + 2:
                del self.plays_rave[(m, s)]
                del self.wins_rave[(m, s)]

    def has_a_winner(self, board):
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        if(len(moved) < self.n_in_row + 2):
            return False, -1

        width = board.width
        height = board.height
        states = board.states
        n = self.n_in_row
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def __str__(self):
        return "AI"
