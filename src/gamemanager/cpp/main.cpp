/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the main UI interface
	This is the main game supervisor used in the ui and simulations for tree search 

	boardSize = since the board is square this is just the width of the board
	score = current player score, starts at 0
	currentBoard = current squares starts at 0 and has all the numbers on the board
	

	AI General Algorithm:
	1. Generate a random init board
	2. loop (in ai using these functions) until 2048
	3. in loop:
		a. guess a move
		b. check new gamestate
		c. retry if fail
		d. go back to a if pass
	4.yay win!
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <string>

/*headers*/
#include "2048.cpp"

using namespace std;

int main(int argc, char *argv[]);

int main(int argc, char *argv[])
{
	//setup
	srand(time(NULL)); //maybe set to a default value for testing

	/*TODO: JMC add commandline arguments like board size and inf play or 2048 play*/
	GameState *currentGame = new GameState(BOARD_SIZE); //setting true will make a random board with an initial number

	//todo: get this to handle any size board !!!!
	currentGame->currentBoard[0][0] = 0;
	currentGame->currentBoard[1][0] = 4;
	currentGame->currentBoard[2][0] = 2;
	currentGame->currentBoard[3][0] = 2;

	while( !determine_2048(currentGame) )
	{
		add_new_number(currentGame);
		print_board(currentGame);

		string* legal_actions = get_legal_actions(currentGame);
		print_legal_actions(legal_actions);

		string action = get_player_action();
		while( !is_action_legal(action, legal_actions) )
		{
			currentGame->invalidMove = true;
			cout << "Invalid action" << endl;
			print_legal_actions(legal_actions);
			action = get_player_action();
		}
		process_action(currentGame, action);

		//print_board(currentGame);
	}

	currentGame->isWon = true;
	print_board(currentGame);
	cout << "Winner!" << endl;
}

