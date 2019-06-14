# NInRow
NInRow-AI. The project aims to reproduce algorithm of AlphaGO in NInRow game, with a low cost of computation, of course.
The entrance script is NInRow.py. The AI can be named Alpha NInRow Rim.



08-06-2018
MCTS-UCT algorithm is implmented. The AI can defeat human player in a 6*6 board with 20000 simulations if given the first move.

09-06-2018
MCTS-UCT algorithm is encapsulated into class MctsUct. Inheritance is implemented. Using a (4-by-4,4-in-row,ai50,ai100) configuration, by simulating 200 games, 
in terms of ai2, the results improve from (p1-135 p2-50 tie-15) to (p1-100 p2-92 tie-8).

12-06-2018
The multi-thread supported training codes are completed. And multi-thread support is also added for the MctsPuct's search. With a CPU of i5-4200H, the seach speed is almost doubled.

14-06-2018
A problem difficult to solve emerged: the network cannot well predict the value/ winning prob of a game. So the search usually goes against the predicted policy. 
However, using our the network ZeroNN, on a 5*5*4 board, with 256 MCTS simulations, Alpha plays 20 games with MCTS-UCT+(both go first 10 games), and wins 15 games while 2 games are ties.

Main References:
1. Mastering the game of Go with deep neural networks and tree search. The paper details method used for AlphaGO Lee, which defeated Lee SeDol in 2016.
2. Mastering the game of Go without human knowledge. The paper details method used for AlphaGO Zeros, a model trained without any human knowledge but is able to defeat AlphaGO Lee.
