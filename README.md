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
Today, progress is made.
Using our the network ZeroNN, on a 5*5*4 board, with 256 MCTS simulations, Alpha-2621 plays 20 games with MCTS-PUCT-(both go first 10 games), and wins 15 games while 2 games are ties.
Here are some denotation:  
  a. Use MCTS-PUCT-[m] to denote MCTS-PUCT using m simulations with P uniformly distributed and V equal to 0. That is, the neural network is not used for search.  
  b. Use Alpha-s[m] to denote AI trained in step s using m simulations for one play.  
  c. Use [r-c-n]-p1-vs-p2-(w,l,t) to denote a game of l-in-row using a r-by-c board and p1 wins w and p2 wins l and t are ties.   
We got Alpha-1037, Alpha-1433, Alpha-2621, Alpha-8561, Alpha-17396， Alpha-39037. Their records against MCTS-UCT+ is (0.7,0.1,0.2), (0.75,0.2,0.05),  (0.75,0.15,0.1), (0.7,0.3,0.0), (0.8,0.2,0.0), (0.75，0.25，0.0).
And Alpha-a can defeat Alpha-b if b > a.
Alpha-1037, 1433, 2621, 8561, 17396, 23761, 29486, 30759, 32672, 37124, 39037, 44765.

15-06-2018
The training is optimized. The latest, maybe not the best, trained Alpha-44793 trained today, playing 4-in-row on a 5*5 board, defeat  
  a. [5-5-4]-(Alpha-44793[256])-vs-(MCTS-PUCT-[256])-(13,6,1),  
  b. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[512])-(14,6,0),  
  c. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[2048])-(9,8,3),  
  d. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[4096])-(13,5,2).  

Main References:
1. Mastering the game of Go with deep neural networks and tree search. The paper details method used for AlphaGO Lee, which defeated Lee SeDol in 2016.
2. Mastering the game of Go without human knowledge. The paper details method used for AlphaGO Zeros, a model trained without any human knowledge but is able to defeat AlphaGO Lee.
