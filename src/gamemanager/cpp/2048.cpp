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

#ifdef CUDA 
    __host__ __device__
#endif
bool determine_2048(GameState *currentGame);

#ifdef CUDA 
    __host__ __device__
#endif
int determine_highest_value(GameState *currentGame);

#ifdef CUDA 
    __host__ __device__
#endif
void print_board(GameState *currentGame);

#ifdef CUDA 
    __host__ __device__
#endif
void print_board(int** currentBoard, int board_size);

#ifdef CUDA 
    __host__ __device__
#endif
void print_horizontal_boarder(int boardSize);

bool add_new_number(GameState *currentGame);

#ifdef CUDA 
    __host__ __device__
#endif
bool checkBoardEmptySlot(GameState *currentGame);

void process_action(GameState *currentGame, int action);
void process_left(GameState *currentGame);
void process_right(GameState *currentGame);
void process_up(GameState *currentGame);
void process_down(GameState *currentGame);

#ifdef CUDA 
    __host__ __device__
#endif
bool compare_game_states(GameState *game1, GameState *game2);

bool determine_2048(GameState *currentGame)
{
	int boardSize = currentGame->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			if (currentGame->currentBoard[i][j] >= 2048)
				return true;
		}
	}

	return false;
}

int determine_highest_value(GameState *currentGame)
{
	int boardSize = currentGame->boardSize;
	int max = 0;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			if (currentGame->currentBoard[i][j] > max)
				max = currentGame->currentBoard[i][j];
		}
	}

	return max;
}

void print_board(int** currentBoard, int board_size)
{
	for (int i = 0; i < board_size; i++)
	{
		print_horizontal_boarder(board_size);

		for (int j = 0; j < board_size; j++)
		{
			printf("| \t%d\t |", currentBoard[i][j]);
		}

		print_horizontal_boarder(board_size);
		printf("\n");
	}
    printf("\n");
}

void print_board(GameState *currentGame)
{
	printf("2048 Game, step: %d\nScore: %d\n\n", currentGame->stepCount, currentGame->score);
	for (int i = 0; i < currentGame->boardSize; i++)
	{
		print_horizontal_boarder(currentGame->boardSize);

		for (int j = 0; j < currentGame->boardSize; j++)
		{
			printf("| \t%d\t |", currentGame->currentBoard[i][j]);
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

bool add_new_number(GameState *currentGame)
{
	// srand(time(NULL)); todo fix this for use_rnd
	int rand_row = rand() % currentGame->boardSize;
	int rand_col = rand() % currentGame->boardSize;

	if(checkBoardEmptySlot(currentGame))
	{
		while(currentGame->currentBoard[rand_row][rand_col] != 0)
		{
			rand_row = rand() % currentGame->boardSize;
			rand_col = rand() % currentGame->boardSize;
		}

		currentGame->currentBoard[rand_row][rand_col] = 2;
		return true;
	}
	return false;
}

bool checkBoardEmptySlot(GameState *currentGame)
{
	int boardSize = currentGame->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			if (currentGame->currentBoard[i][j] == 0)
				return true;
		}
	}

	return false;
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
					modified[t+1] = true;
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

bool compare_game_states(GameState *game1, GameState *game2)
{
	int boardSize = game1->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			if (game1->currentBoard[i][j] != game2->currentBoard[i][j])
				return false;
		}
	}

	return true;
}