#include "../gamemanager/cpp/2048.cpp"

	/*"left",	//0
      "right",	//1
      "up",		//2
      "down"	//3*/

class Node 
{
	private:
		Node *node_left;
		Node *node_right;
		Node *node_up;
		Node *node_down;

	public:
		GameState* current_state = NULL;
		Node* parent = NULL;

		Node* children[4] = {node_left, node_right, node_up, node_down}; // new Node[4];

		int depth = 0;

		bool test = false;

		Node();
		Node(Node*, GameState*, int);
		// ~Node();
		// void Node::AddNode(Node* current_node, GameState* state, int direction);
};

Node::Node()
{

}

Node::Node(Node* _parent, GameState* state, int _depth)
{
	parent = _parent;
	current_state = state;
	depth = _depth;
}

// Node::~Node()
// {
// 	delete[] this->children;
// 	free(this->current_state);
// 	this->parent = NULL;
// }

/*void Node::AddNode(Node* current_node, GameState* state, int direction)
{
	int currentDepth = current_node -> depth + 1;
	if(this->max_depth < currentDepth)
		this->max_depth = currentDepth;

	Node* node = new Node(current_node, state, currentDepth);
	current_node->children[direction] = *node;

	num_nodes++;
}*/