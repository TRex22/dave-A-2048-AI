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
		Node* optimal2048 = NULL;
		int num_solutions = 0;
		int num_leaves = 0;
		int num_cutoff_states = 0;

		Tree(GameState* state);
		~Tree();
		
};

Tree::Tree(GameState* state)
{
	this->BOARD_SIZE = state->boardSize;
	Node* parent;
	Node* head = new Node(parent, state, 0);
    head -> isLeaf = false; // root is assumed never to be a leaf, not needed tho
	this->root = head;
}

Tree::~Tree()
{
	delete root;
	delete &BOARD_SIZE;
	delete &num_nodes;
	delete &max_depth;
	delete &num_solutions;
	delete &num_leaves;
	delete &num_cutoff_states;
}

