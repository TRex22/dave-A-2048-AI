#include "../gamemanager/cpp/2048.cpp"
const int csize = 4;

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

		Node* children[csize] = {node_left, node_right, node_up, node_down}; 
		int depth = 0;

		bool isLeaf = false;
        bool hasChildren = false;

		Node(Node*, GameState*, int);
		~Node();
};

Node::Node(Node* _parent, GameState* state, int _depth)
{
	parent = _parent;
	current_state = state;
	depth = _depth;
}

Node::~Node()
{
	delete children[0];
	delete children[1];
	delete children[2];
	delete children[3];

	delete current_state;
	delete &depth;
	delete &isLeaf;
	delete &hasChildren;
}