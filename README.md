Dave A 2048 AI
==============

This is an AI built on Cuda and MPI.
It uses tree search algorithms to play the single player game 2048.

2048
----


TODO
----

- ~~cleanup comments~~
- save all moves that have been calculated to viz later
- save initial game state
- open initial game state to use
- Visulizer
- Compare two moves files so we can check two ais
- Maybe make sonething that can run through a set of moves and check it makes sense? - probably not just - watch the vid
- test runner
- change string to int

Gamemanager
- fix main file
- ~~add stopping condition~~
- ~~gamestate must contain legal moves internally so we can return it.~~
- ~~get this working as a way for the ai to play the game i.e. a function which is the game and allows the ai to inject a move.~~
- ~~fix printout to only show one board per state ... its also a bit weird (might be ssh)~~
- ~~setup random game state to make proper results~~
- add commandline arguments like board size
- get this to handle any size board !!!!
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

References
----
- http://kawahara.ca/how-to-run-an-ipythonjupyter-notebook-on-a-remote-machine/

- http://robots.stanford.edu/cs221/2016/restricted/projects/prithvir/final.pdf
- http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048
- https://news.ycombinator.com/item?id=7379821

- http://stackoverflow.com/questions/5278580/non-recursive-depth-first-search-algorithm
- https://dev.to/vaidehijoshi/demystifying-depth-first-search
- http://www.programming-techniques.com/2012/07/depth-first-search-in-c-algorithm-and.html
- https://en.wikipedia.org/wiki/Depth-first_search
