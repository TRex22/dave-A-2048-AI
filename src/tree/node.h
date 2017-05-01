#include "../gamemanager/cpp/2048.cpp"

class Node 
{
	private:
		Node *node_left;
		Node *node_right;
		Node *node_up;
		Node *node_down;
		Node *parent_tmp;

	public:
		GameState* current_state = NULL;
		Node* parent = parent_tmp;

		Node* children[4] = {node_left, node_right, node_up, node_down}; 
		int depth = 0;

		bool test = false;

		Node(Node&, GameState*, int);
};

Node::Node(Node &_parent, GameState* state, int _depth)
{
	parent = &_parent;
	current_state = state;
	depth = _depth;
}