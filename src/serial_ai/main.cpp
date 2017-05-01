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
bool canContinue(Node* node);
bool isLeaf(Node* node);
void buildTree(Tree* tree);
void findChildren(Tree* tree, Node* node);
Node* findCurrentNodeFromChildren(Node* current_node);

int main(int argc, char *argv[])
{
	srand(time(NULL));
	GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);

	buildTree(tree);

	printf("%d, %d\n", tree->num_nodes, tree->max_depth);

}

bool checkAtRoot(Node* node)
{
	if (node->depth == 0)
		return true;
	return false;
}

bool canContinue(Node* node)
{
	bool won = determine_2048(node->current_state);

	if(!checkAtRoot(node) && !won)
		return true;

	if(node->test == false)
		return true;

	return false;
}

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

Node* findCurrentNodeFromChildren(Node* current_node)
{
	Node* new_current_node;

	if(current_node->children[0]->test == false)
	{
		new_current_node = (current_node->children[0]);
	}
	else if(current_node->children[1]->test == false)
	{
		new_current_node = current_node->children[1];
	}
	else if(current_node->children[2]->test == false)
	{
		new_current_node = current_node->children[2];
	}
	else if(current_node->children[3]->test == false)
	{
		new_current_node = current_node->children[3];
	}
	else
	{
		new_current_node = NULL;
	}
	
	return new_current_node;
}

void buildTree(Tree* tree)
{
	Node* root = tree->root;
	Node* current_node = root;
	Node* parent_node = current_node;

	while(canContinue(current_node))
	{
		findChildren(tree, current_node);
		// printf("%d, %d\n", current_node->parent->children, 0);

		Node* new_current_node = findCurrentNodeFromChildren(current_node);

		while(new_current_node == NULL && !checkAtRoot(new_current_node))
		{
			new_current_node = findCurrentNodeFromChildren(parent_node->parent);
			parent_node = parent_node->parent;
		}

		current_node = new_current_node;
	}
}

void findChildren(Tree* tree, Node* node)
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
		add_new_number(states[i]);

		int currentDepth = node->depth + 1;
		if(tree->max_depth < currentDepth)
			tree->max_depth = currentDepth;
		
		node->children[i] = new Node(node, states[i], currentDepth);

		tree->num_nodes++;
	}

	node->test = true;
	printf("%d, %d, %d\n", tree->num_nodes, tree->max_depth, determine_highest_value(node->current_state));
}
