#include <stack>

#include "../tree/tree.h"

using namespace std;

Tree* buildTree(Tree* tree);
void generateChidlren(Node* currentNode, stack<Node> tracker);
bool checkAtRoot(Node* node);
bool canContinue(Node* node);
bool isLeaf(Node* node);

Tree* buildTree(Tree* tree)
{
	stack<Node> tracker;
	tracker.push(tree.root);

	while(!tracker.empty())
	{
		int board_size = 4; //TODO: make as parameter to do this properly
		Node* currentNode = tracker.pop();

		if(!leaf)
		{
			generateChidlren(currentNode, tracker, board_size);
		}

	}
}

void generateChidlren(Node* currentNode, stack<Node> tracker, int board_size)
{
	GameState *state_left;
	GameState *state_right;
	GameState *state_up;
	GameState *state_down;

	GameState* states[] = {state_left, state_right, state_up, state_down};

	for (int i = 3; i >= 0; --i)
	{
		states[i] = new GameState(board_size);
		states[i]->copy(currentNode.current_state);
		process_action(states[i], i);
		add_new_number(states[i]);

		int currentDepth = node.depth + 1;
		if(tree->max_depth < currentDepth)
			tree->max_depth = currentDepth;
		
		currentNode.children[i] = new Node(currentNode, states[i], currentDepth);

		tree->num_nodes++;
	}

	currentNode.test = true;
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