/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the main UI interface
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <time.h>
#include <omp.h>
#include <string>

/*headers*/
#include "../headers/main.h"
#include "../headers/2048.h"

/*Global Variables and Definitions*/
#define BOARD_SIZE 4

using namespace std;

int main(int argc, char *argv[])
{
	/*TODO: JMC add commandline arguments like board size*/

	
}

void print_board(2048GameState currentGame)
{
	printf("2048 Game, step: %d\nScore: %d\n\n", currentGame.stepCount, currentGame.score);
	for (int i = 0; i < boardSize; i++)
	{
		print_horizontal_boarder(boardSize);

		for (int j = 0; j < boardSize; j++)
		{
			printf("| %d |", currentGame.currentBoard[i][j]);
		}

		print_horizontal_boarder(boardSize);
	}
}

void print_horizontal_boarder(int boardSize)
{
	for (int k = 0; k < boardSize; k++)
	{
		printf("-"); //this is to make a boarder
	}
	printf("\n");
}