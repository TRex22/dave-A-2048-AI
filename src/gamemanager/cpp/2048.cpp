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

#include "../headers/2048.h"

using namespace std;


/*Global Variables and Definitions*/
// not using define in order to overwrite with cmdline args at a later point
int BOARD_SIZE = 4;

// GameState* run_gamestate(GameState currentGame, bool print_game, int move);
bool determine_2048(GameState *currentGame);
void print_board(GameState *currentGame);
void print_horizontal_boarder(int boardSize);
void add_new_number(GameState *currentGame);
// string get_player_action();
void process_action(GameState *currentGame, int action);
void process_left(GameState *currentGame);
void process_right(GameState *currentGame);
void process_up(GameState *currentGame);
void process_down(GameState *currentGame);
// int* get_legal_actions(GameState *currentGame);
// int count_actions(int* actions);
// void print_legal_actions(string* legal_actions);
// bool is_action_legal(int action, int* legal_actions);

//all moves as index 0 -> 3 for the ai
const string Moves[]
{
    "left", //0
    "right", //1
    "up", //2
    "down" //3
};



//todo: gamestate must contain legal moves internally so we can return it.
// GameState* run_gamestate(GameState *currentGame, bool print_game, int move) //todo boardsize etc ...
// {
// 	if(currentGame == NULL)
// 		exit(EXIT_FAILURE); //just a reminder for myself

// 	GameState *newGame;
// 	newGame->copy(currentGame); //slow maybe? TODO!!

// 	string* legal_actions = get_legal_actions(currentGame);
// 	string action = Moves[move];

// 	bool check_legal_move = is_action_legal(action, legal_actions);

// 	if(check_legal_move)
// 	{
// 		process_action(currentGame, action);
// 		string* legal_actions = get_legal_actions(currentGame);
// 	}

// 	currentGame->invalidMove = !check_legal_move;

// 	return currentGame;
// }

bool determine_2048(GameState *currentGame)
{
	int boardSize = currentGame->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			if (currentGame->currentBoard[i][j] == 2048)
				return true;
		}
	}

	return false;
}

void print_board(GameState *currentGame)
{
	printf("2048 Game, step: %d\nScore: %d\n\n", currentGame->stepCount, currentGame->score);
	for (int i = 0; i < currentGame->boardSize; i++)
	{
		print_horizontal_boarder(currentGame->boardSize);

		for (int j = 0; j < currentGame->boardSize; j++)
		{
			printf("| %d |", currentGame->currentBoard[i][j]);
		}

		print_horizontal_boarder(currentGame->boardSize);
		printf("\n");
	}
}

void print_horizontal_boarder(int boardSize)
{
	for (int k = 0; k < boardSize; k++)
	{
		printf("-"); //this is to make a boarder
	}
}

void add_new_number(GameState *currentGame)
{
	 int rand_row = rand() % currentGame->boardSize;
	 int rand_col = rand() % currentGame->boardSize;

	 while(currentGame->currentBoard[rand_row][rand_col] != 0)
	 {
	 	rand_row = rand() % currentGame->boardSize;
	 	rand_col = rand() % currentGame->boardSize;
	 }

	 currentGame->currentBoard[rand_row][rand_col] = 2;
}

string get_player_action()
{
	string str;
	cout << "Input action: ";
	cin >> str;
	return str;
}

void process_action(GameState *currentGame, int action)
{
	if (action == 0)
	{
		process_left(currentGame);
	}
	else if (action == 1)
	{
		process_right(currentGame);
	}
	else if (action == 2)
	{
		process_up(currentGame);
	}
	else if (action == 3)
	{
		process_down(currentGame);
	}
}



void process_left(GameState *currentGame)
{
	for (int i = 0; i < currentGame->boardSize; ++i)
	{
		bool modified[currentGame->boardSize];
		for (int p = 0; p < currentGame->boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int j = 1; j < currentGame->boardSize; ++j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t > 0 && currentGame->currentBoard[i][t-1] == 0)
				{
					currentGame->currentBoard[i][t-1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[i][t-1] == currentGame->currentBoard[i][t] &&  modified[t-1] == false)
				{
					currentGame->currentBoard[i][t-1] += currentGame->currentBoard[i][t];
					modified[t-1] = true;
					currentGame->score += currentGame->currentBoard[i][t-1];
					currentGame->currentBoard[i][t] = 0;
				}
			}
			
		}
	}
}

void process_right(GameState *currentGame)
{
	for (int i = 0; i < currentGame->boardSize; ++i)
	{
		bool modified[currentGame->boardSize];
		for (int p = 0; p < currentGame->boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int j = currentGame->boardSize - 2; j > -1; --j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t < currentGame->boardSize - 1 && currentGame->currentBoard[i][t+1] == 0)
				{
					currentGame->currentBoard[i][t+1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t++;
				}

				if (t == currentGame->boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[i][t+1] == currentGame->currentBoard[i][t] && modified[t+1] == false)
				{
					currentGame->currentBoard[i][t+1] += currentGame->currentBoard[i][t];
					modified[t+1] = true;
					currentGame->score += currentGame->currentBoard[i][t+1];
					currentGame->currentBoard[i][t] = 0;
				}
			}
		}
	}
}

void process_up(GameState *currentGame)
{
	for (int j = 0; j < currentGame->boardSize; ++j)
	{
		bool modified[currentGame->boardSize];
		for (int p = 0; p < currentGame->boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int i = 1; i < currentGame->boardSize; ++i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t > 0 && currentGame->currentBoard[t-1][j] == 0)
				{
					currentGame->currentBoard[t-1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[t-1][j] == currentGame->currentBoard[t][j] &&  modified[t-1] == false)
				{
					currentGame->currentBoard[t-1][j] += currentGame->currentBoard[t][j];
					modified[t+1] == true;
					currentGame->score += currentGame->currentBoard[i][t-1];
					currentGame->currentBoard[t][j] = 0;
				}
			}
		}
	}
}

void process_down(GameState *currentGame)
{
	for (int j = 0; j < currentGame->boardSize; ++j)
	{
		bool modified[currentGame->boardSize];
		for (int p = 0; p < currentGame->boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int i = currentGame->boardSize - 2; i > -1; --i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t < currentGame->boardSize - 1 && currentGame->currentBoard[t+1][j] == 0)
				{
					currentGame->currentBoard[t+1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t++;
				}

				if (t == currentGame->boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[t+1][j] == currentGame->currentBoard[t][j] && modified[t+1] == false)
				{
					currentGame->currentBoard[t+1][j] += currentGame->currentBoard[t][j];
					modified[t+1] = true;
					currentGame->score += currentGame->currentBoard[t+1][j];
					currentGame->currentBoard[t][j] = 0;
				}
			}
		}
	}
}

// int* get_legal_actions(GameState *currentGame)
// {
// 	int all_actions[] = {0, 1, 2, 3};
// 	int* legal_actions[4] = { NULL };

// 	GameState *state_left;
// 	GameState *state_right;
// 	GameState *state_up;
// 	GameState *state_down;

// 	GameState* states[] = {state_left, state_right, state_up, state_down};

// 	for (int i = 0; i < 4; ++i)
// 	{
// 		states[i] = new GameState(BOARD_SIZE);
// 		states[i]->copy(currentGame);
// 		process_action(states[i], all_actions[i]);
// 		if ( !currentGame->equals(states[i]) )
// 		{
// 			legal_actions[i] = all_actions[i];
// 		}
// 	}

// 	return legal_actions;
// }

// int count_actions(int* actions)
// {
// 	if (actions[0] == NULL && actions[1] == NULL && actions[2] == NULL && actions[3] == NULL)
// 	{
// 		return 0;
// 	}

// 	int count = 0;
// 	for (int i = 0; i < 4; ++i)
// 	{
// 		if ( actions[i] != NULL )
// 		{
// 			count++;
// 		}
// 	}

// 	return count;
// }

// void print_legal_actions(string* legal_actions)
// {
// 	cout << "Legal actions are: ";
// 	for (int i = 0; i < 4; ++i)
// 	{
// 		if (!legal_actions[i].empty())
// 		{
// 			cout << legal_actions[i] << " ";
// 		}
// 	}
// 	cout << endl;
// }

// bool is_action_legal(int action, int* legal_actions)
// {
// 	bool result = false;

// 	for (int i = 0; i < 4; ++i)
// 	{
// 		if (legal_actions[i] == action)
// 		{
// 			result = true;
// 			cout << action << " is " << legal_actions[i] << endl; 
// 		}
// 	}

// 	return result;
// }