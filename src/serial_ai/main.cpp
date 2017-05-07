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
void print_left_most_path(Tree* tree);
bool print_solution(Tree* tree);

int main(int argc, char *argv[])
{
	srand(time(NULL));
	int i = board_size; //todo fix this
	GameState* initial_state = new GameState(i);
	add_new_number(initial_state);

	initial_state->currentBoard[0][0] = 1024;
	initial_state->currentBoard[0][1] = 1024;

	Tree* tree = new Tree(initial_state);
	buildTree_with_ustack(tree, -1, 20000);

	printf("%i: num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", 
			i, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));

	// printf("%d, %d, sols: %d, leaves: %d, stats: %f\n", tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));
}

void print_left_most_path(Tree* tree)
{
	Node* node = tree->root;
    while (node)
    { 
        print_board(node->current_state);

        node = node->children[1];
    }
}

bool print_solution(Tree* tree)
{
	if(tree->a2048)
    {
    	Node* node = tree->a2048;
    	while(node && !checkAtRoot(node))
    	{
    		print_board(node->current_state);
    		node = node->parent;
    	}
    	print_board(node->current_state);
    }
    else
    	printf("NULL@@@@@@\n");
    return true;
}


