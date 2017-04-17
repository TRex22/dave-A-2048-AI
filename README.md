Dave A 2048 AI
==============

This is an AI built on Cuda and MPI.
It uses tree search algorithms to play the single player game 2048.

2048
----


TODO
----

Gamemanager
- add stopping condition
- gamestate must contain legal moves internally so we can return it.
- get this working as a way for the ai to play the game i.e. a function which is the game and allows the ai to inject a move.
- add commandline arguments like board size
- get this to handle any size board !!!!
- fix printout to only show one board per state ... its also a bit weird (might be ssh)

Serial Ai Tree Search
- Hook into game and play it as demo
- Better Demo?

Cuda Ai Search
- Show increased size for less time?

MPI Ai Search
- ????? ;(