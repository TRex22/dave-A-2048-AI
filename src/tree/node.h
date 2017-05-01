#include "../gamemanager/cpp/2048.cpp"

	/*"left",	//0
      "right",	//1
      "up",		//2
      "down"	//3*/

class Node 
{
	private:
		

	public:
		GameState* current_state = NULL;
		Node* parent = NULL;

		Node* children = new Node[4];

		// Node* left = NULL;
		// Node* right = NULL;
		// Node* up = NULL;
		// Node* down = NULL;
		int depth = 0;

		Node();
		Node(Node*, GameState*, int);
		~Node();
		void DeleteLeft();
		void DeleteRight();
		void DeleteUp();
		void DeleteDown();
		// void Node::AddNode(Node* current_node, GameState* state, int direction);
};

Node::Node()
{
	
}

Node::Node(Node* parent, GameState* state, int _depth)
{
	this->parent = parent;
	this->current_state = state;
	depth = _depth;
}

Node::~Node()
{
	delete[] this->children;
	free(this->current_state);
	this->parent = NULL;
}

/*void Node::AddNode(Node* current_node, GameState* state, int direction)
{
	int currentDepth = current_node -> depth + 1;
	if(this->max_depth < currentDepth)
		this->max_depth = currentDepth;

	Node* node = new Node(current_node, state, currentDepth);
	current_node->children[direction] = *node;

	num_nodes++;
}*/