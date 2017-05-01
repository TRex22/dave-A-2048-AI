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

#include "../tree/tree.h"

int board_size = 4;

int main(int argc, char *argv[]);
bool checkAtRoot(Node* node);
// bool canContinue(Node* node);
bool isLeaf(Node* node);
void buildTree(Tree* tree, Node* node);

int main(int argc, char *argv[])
{
	srand(time(NULL));
	GameState* initial_state = new GameState(board_size);

	Tree* tree = new Tree(initial_state);

	buildTree(tree, tree->root);

	printf("%d, %d\n", tree->num_nodes, tree->max_depth);

}

bool checkAtRoot(Node* node)
{
	if (node->depth == 0)
		return true;
	return false;
}

// bool canContinue(Node* node)
// {
// 	if(!checkAtRoot(node))
// 		return true;

// 	if(node->children[0] != NULL && node->children[1] != NULL && node->children[2] != NULL && node->children[3] != NULL)
// 		return false;

// 	return true;
// }

bool isLeaf(Node* node)
{
	GameState *state_left;
	GameState *state_right;
	GameState *state_up;
	GameState *state_down;

	GameState* states[] = {state_left, state_right, state_up, state_down};

	for (int i = 0; i < 4; ++i)
	{
		states[i] = new GameState(board_size);
		states[i]->copy(node->current_state);
		process_action(states[i], i);
		if ( node->current_state->equals(states[i]) )
		{
			return false;
		}
	}

	return true;
}

void buildTree(Tree* tree, Node* node)
{
	if(!isLeaf(node))
	{
		GameState *state_left;
		GameState *state_right;
		GameState *state_up;
		GameState *state_down;

		GameState* states[] = {state_left, state_right, state_up, state_down};

		for (int i = 0; i < 4; ++i)
		{
			states[i] = new GameState(board_size);
			states[i]->copy(node->current_state);
			process_action(states[i], i);

			int currentDepth = node->depth + 1;
			if(tree->max_depth < currentDepth)
				tree->max_depth = currentDepth;
			
			Node* node = new Node(node, states[i], currentDepth);
			node->children[i] = *node;

			tree->num_nodes++;
		}
	}

	buildTree(tree, &node->children[0]);
	buildTree(tree, &node->children[1]);
	buildTree(tree, &node->children[2]);
	buildTree(tree, &node->children[3]);

}
