/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the main game supervisor used in the ui and simulations of MCTS

	boardSize = since the board is square this is just the width of the board
	score = current player score, starts at 0
	currentBoard = current squares starts at 0 and has all the numbers on the board
	stepCount = 
*/

using namespace std;
#include "../headers/2048.h"

class 2048GameState {
private:
	

public:
	int boardSize;
	int score = 0;
	int* currentBoard;
	int stepCount = 0;


}