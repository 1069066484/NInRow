# NInRow
NInRow-AI. The project aims to reproduce algorithm of AlphaGO in NInRow game, with a low cost of computation, of course.
The entrance script is NInRow.py.



08-06-2018
MCTS-UCT algorithm is implmented. The AI can defeat human player in a 6*6 board with 20000 simulations if given the first move.

09-06-2018
MCTS-UCT algorithm is encapsulated into class MctsUct. Inheritance is implemented. Using a (4-by-4,4-in-row,ai50,ai100) configuration, by simulating 200 games, 
in terms of ai2, the results improve from (p1-135 p2-50 tie-15) to (p1-100 p2-92 tie-8).


Main References:
1. Mastering the game of Go with deep neural networks and tree search. The paper details method used for AlphaGO Lee, which defeated Lee SeDol in 2016.
2. Mastering the game of Go without human knowledge. The paper details method used for AlphaGO Zeros, a model trained without any human knowledge.
