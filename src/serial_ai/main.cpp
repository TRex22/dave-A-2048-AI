/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the serial AI approach

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../tree_builder/tree_builder.cpp"

const int board_size = 4;

int main(int argc, char *argv[]);


int main(int argc, char *argv[])
{
	srand(time(NULL));
	GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);
	// printf("%d\n", tree->root);
    buildTree_inplace(tree, -1, 10000);
	// buildTree_with_ustack(tree, 231);
    // buildTree_with_ustack(tree, -1, 10000);
	printf("%d, %d\n", tree->num_nodes, tree->max_depth);

}


