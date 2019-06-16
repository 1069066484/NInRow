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

16-06-2018
Here are some denotation:  
  &ensp; a. Use MCTS-PUCT-[m] to denote MCTS-PUCT using m simulations with P uniformly distributed and V equal to 0. That is, the neural network is not used for search.  
  &ensp; b. Use Alpha-s[m] to denote AI trained in step s using m simulations for one play.  
  &ensp; c. Use [r-c-n]-p1-vs-p2-(w,l,t) to denote a game of l-in-row using a r-by-c board and p1 wins w and p2 wins l and t are ties. 
An AI that plays 4-in-row on a 5*5 board is trained.

15-06-2018
The training is optimized. The latest, maybe not the best, trained Alpha-44793 trained today, playing 4-in-row on a 5*5 board, got great performance.    
  &ensp; a. [5-5-4]-(Alpha-44793[256])-vs-(MCTS-PUCT-[256])-(13,6,1),  
  &ensp; b. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[512])-(14,6,0),  
  &ensp; c. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[2048])-(9,8,3),  
  &ensp; d. [5-5-4]-(Alpha-44793[512])-vs-(MCTS-PUCT-[4096])-(13,5,2).  

Given the first move, AI can play well.
![Image text](https://github.com/1069066484/NInRow/blob/master/ai554_6_16/5.png)

Main References:
1. Mastering the game of Go with deep neural networks and tree search. The paper details method used for AlphaGO Lee, which defeated Lee SeDol in 2016.
2. Mastering the game of Go without human knowledge. The paper details method used for AlphaGO Zeros, a model trained without any human knowledge but is able to defeat AlphaGO Lee.
