# NInRow
NInRow-AI. The project aims to reproduce algorithm of AlphaGO in NInRow game, with a low cost of computation, of course.
The entrance script is NInRow.py. 


Main References:
1. Mastering the game of Go with deep neural networks and tree search. The paper details method used for AlphaGO Lee, which defeated Lee SeDol in 2016.
2. Mastering the game of Go without human knowledge. The paper details method used for AlphaGO Zeros, a model trained without any human knowledge but is able to defeat AlphaGO Lee.

-----------------------------------------------------------------------------------------------
三年前的代码了，今天维护一下readme，代码实现的是2017年的AlphaGoZero。
虽然我很喜欢alphaGo，但是这东西太吃算力了，写这个当作学习一下吧。
我最后训练得较好的是6x6的棋盘上的四子棋，但是没有搜到必胜策略。
用的是tensorflow，这也许是我用tf写的最后一个项目了。
下面阐述一下各个文件夹实现的内容：
- mcts：实现了蒙特卡洛搜索算法。
- nets：实现了网络结构。
- trainer：实现了训练器。
- trans：这里是一些我自己的备份代码，是最简化的AGZ实现。
- games：这里实现了游戏逻辑。
- NInRow.py：启动游戏。
