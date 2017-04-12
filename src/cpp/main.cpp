/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the main UI interface
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
#include "../headers/main.h"
#include "../headers/2048.h"

/*Global Variables and Definitions*/
#define BOARD_SIZE 4

using namespace std;

int main(int argc, char *argv[]);
void print_board(GameState *currentGame);
void print_horizontal_boarder(int boardSize);
void add_new_number(GameState *currentGame);
string get_player_action();
void process_action(GameState *currentGame, string action);
void process_left(GameState *currentGame);
void process_right(GameState *currentGame);
void process_up(GameState *currentGame);
void process_down(GameState *currentGame);

int main(int argc, char *argv[])
{
	/*TODO: JMC add commandline arguments like board size*/
	GameState *currentGame = new GameState(BOARD_SIZE);
	currentGame->currentBoard[0][0] = 0;
	currentGame->currentBoard[1][0] = 4;
	currentGame->currentBoard[2][0] = 2;
	currentGame->currentBoard[3][0] = 8;
	currentGame->currentBoard[3][1] = 4;

	while(1){
		add_new_number(currentGame);
		print_board(currentGame);
		string action = get_player_action();
		process_action(currentGame, action);
		print_board(currentGame);
	}
	



	
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

void add_new_number(GameState *currentGame){
	 int rand_row = rand() % currentGame->boardSize;
	 int rand_col = rand() % currentGame->boardSize;

	 while(currentGame->currentBoard[rand_row][rand_col] != 0){
	 	rand_row = rand() % currentGame->boardSize;
	 	rand_col = rand() % currentGame->boardSize;
	 }

	 currentGame->currentBoard[rand_row][rand_col] = 2;
}

string get_player_action(){
	string str;
	cout << "Input action: ";
	cin >> str;
	return str;
}

void process_action(GameState *currentGame, string action){
	if (action == "left" || action == "l")
	{
		process_left(currentGame);
	}
	else if (action == "right" || action == "r")
	{
		process_right(currentGame);
	}
	else if (action == "up" || action == "u")
	{
		process_up(currentGame);
	}
	else if (action == "down" || action == "d")
	{
		process_down(currentGame);
	}
}

void process_left(GameState *currentGame){
	for (int i = 0; i < currentGame->boardSize; ++i)
	{
		for (int j = 1; j < currentGame->boardSize; ++j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t > 0 && currentGame->currentBoard[i][t-1] == 0){
					currentGame->currentBoard[i][t-1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[i][t-1] == currentGame->currentBoard[i][t])
				{
					currentGame->currentBoard[i][t-1] += currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
				}
			}
			
		}
	}
}

void process_right(GameState *currentGame){
	for (int i = 0; i < currentGame->boardSize; ++i)
	{
		for (int j = currentGame->boardSize - 2; j > -1; --j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t < currentGame->boardSize - 1 && currentGame->currentBoard[i][t+1] == 0){
					currentGame->currentBoard[i][t+1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t++;
				}

				if (t == currentGame->boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[i][t+1] == currentGame->currentBoard[i][t])
				{
					currentGame->currentBoard[i][t+1] += currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
				}
			}
		}
	}
}

void process_up(GameState *currentGame){
	for (int j = 0; j < currentGame->boardSize; ++j)
	{
		for (int i = 1; i < currentGame->boardSize; ++i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t > 0 && currentGame->currentBoard[t-1][j] == 0){
					currentGame->currentBoard[t-1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[t-1][j] == currentGame->currentBoard[t][j])
				{
					currentGame->currentBoard[t-1][j] += currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
				}
			}
		}
	}
}

void process_down(GameState *currentGame){
	for (int j = 0; j < currentGame->boardSize; ++j)
	{
		for (int i = currentGame->boardSize - 2; i > -1; --i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t < currentGame->boardSize - 1 && currentGame->currentBoard[t+1][j] == 0){
					currentGame->currentBoard[t+1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t++;
				}

				if (t == currentGame->boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[t+1][j] == currentGame->currentBoard[t][j])
				{
					currentGame->currentBoard[t+1][j] += currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
				}
			}

			//cout << currentGame->currentBoard[i][j] << ",";
		}
	}
}