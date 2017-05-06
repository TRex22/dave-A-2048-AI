#include "node.h"
#include <stdlib.h>

using namespace std;

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
		
};

Tree::Tree(GameState* state)
{
	this->BOARD_SIZE = state->boardSize;
	Node* parent;
	Node* head = new Node(parent, state, 0);
    head -> isLeaf = false; // root is assumed never to be a leaf, not needed tho
	this->root = head;
}

