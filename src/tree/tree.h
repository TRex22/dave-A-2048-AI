// #include "../gamemanager/headers/2048.h"
#include "node.h"
#include <stdlib.h>

using namespace std;

/*	  "left",	//0
      "right",	//1
      "up",		//2
      "down"	//3*/

class Tree 
{
	private:
		

	public:
		int BOARD_SIZE = 0;

		Node* root = NULL;
		int num_nodes = 0;
		int max_depth = 0;
		/*int min_depth = 0;*/

		Tree(GameState* state);
		// void AddNode(Node* current_node, GameState* state, int direction);
		
};

Tree::Tree(GameState* state)
{
	this->BOARD_SIZE = state->boardSize;
	Node* head = new Node(NULL, state, 0);
	this->root = head;
}

// void Tree::AddNode(Node* current_node, GameState* state, int direction)
// {
// 	int currentDepth = current_node -> depth + 1;
// 	if(this->max_depth < currentDepth)
// 		this->max_depth = currentDepth;

// 	Node* node = new Node(current_node, state, currentDepth);
// 	current_node->children[direction] = *node;

// 	num_nodes++;
// }

