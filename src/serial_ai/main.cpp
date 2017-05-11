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
    
    input:
        board size (b)
        lock rnd flag
        max depth (d)
        max nodes (n)
        save to file (f)
        print_cmd (p) t/f
        initial state (is)
        debug (d)
        
    timing
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stack>
#include "omp.h"

#include "../helper/helper.h"

const int board_size = 4;
bool DEBUG = true;

int main(int argc, char *argv[]);
Tree* buildTree(Tree* tree, int depth_limit, int node_limit);
void generateChidlren(Node* currentNode, Tree* tree);

int main(int argc, char *argv[])
{
	srand(time(NULL));
	int i = board_size; //todo fix this
	GameState* initial_state = new GameState(i);
	add_new_number(initial_state);

	initial_state->currentBoard[0][0] = 1024;
	initial_state->currentBoard[0][1] = 1024;

	Tree* tree = new Tree(initial_state);
	buildTree(tree, -1, 20000);

	printf("%i: num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", 
			i, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));
}

Tree* buildTree(Tree* tree, int depth_limit = -1, int node_limit = -1)
{
    //todo: fix this not working correctly
	stack<Node*> tracker;
	tracker.push(tree->root);

	while(!tracker.empty() && !shouldLimit(tree, depth_limit, node_limit))
	{
		Node* currentNode = tracker.top();
        tracker.pop();

		if(currentNode)
		{
			generateChidlren(currentNode, tree);
            
            for (int i = 3; i > -1; --i)
            {
                tracker.push(currentNode->children[i]);
            }
		}
        
        if(DEBUG)
        {
            // printf("%li\n", tracker.size());
        }
	}
}

void generateChidlren(Node* currentNode, Tree* tree)
{
	for (int i = 0; i < 4; i++)
	{
        GameState* newState = new GameState(tree->BOARD_SIZE);
		newState->copy(currentNode->current_state);

		process_action(newState, i);

        if(!determine_2048(currentNode->current_state) && !compare_game_states(currentNode->current_state, newState))
        {
            bool fullBoard = !add_new_number(newState);
            if(!fullBoard)
            {
                int currentDepth = currentNode->depth + 1;
                if(tree->max_depth < currentDepth)
                    tree->max_depth = currentDepth;
                
                currentNode->children[i] = new Node(currentNode, newState, currentDepth);
                tree->num_nodes++;

                currentNode->hasChildren = true;
            }
            else
            {
                currentNode->children[i] = nullptr;
            }
        }
        else
        {
            currentNode->children[i] = nullptr;
        }

        if(determine_2048(currentNode->current_state)) //win
        {
            tree->a2048 = currentNode;
            tree->num_solutions++;
        }

        if(determine_2048(currentNode->current_state) || compare_game_states(currentNode->current_state, newState)) //leaf
        {
            tree->num_leaves++;
        }
	}
    
    if(DEBUG)
    {
        // printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        // print_board(currentNode->current_state);
    }
}


