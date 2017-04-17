Dave A 2048 AI
==============

This is an AI built on Cuda and MPI.
It uses tree search algorithms to play the single player game 2048.

2048
----


TODO
----

Gamemanager
- ~~add stopping condition~~
- ~~gamestate must contain legal moves internally so we can return it.~~
- ~~get this working as a way for the ai to play the game i.e. a function which is the game and allows the ai to inject a move.~~
- add commandline arguments like board size
- get this to handle any size board !!!!
- ~~fix printout to only show one board per state ... its also a bit weird (might be ssh)~~
- ~~setup random game state to make proper results ... ~~
- use better prng maybe mt19937_64

Serial Ai Tree Search
- Hook into game and play it as demo
- Better Demo?
- make a tree from the gamestates
- search tree until 2048
- have backtrack
- pick an algorithm to use (I prefer MCTS due to using simulations for the tree. Minimax may make the tree too intractable at 8x8 board size)

Cuda Ai Search
- Show increased size for less time?

MPI Ai Search
- ????? ;(