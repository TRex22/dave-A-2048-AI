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

		Node* left = NULL;
		Node* right = NULL;
		Node* up = NULL;
		Node* down = NULL;
		int depth = 0;

		Node(Node*, GameState*, int);
		~Node();
		void DeleteLeft();
		void DeleteRight();
		void DeleteUp();
		void DeleteDown();
};

Node::Node(Node* parent, GameState* state, int _depth)
{
	this->parent = parent;
	this->current_state = state;
	depth = _depth;
}

Node::~Node()
{
	free(this->left);
	free(this->right);
	free(this->up);
	free(this->down);
	free(this->current_state);
	this->parent = NULL;
}

void Node::DeleteLeft()
{
	free(this->left);
}

void Node::DeleteRight()
{
	free(this->right);	
}

void Node::DeleteUp()
{
	free(this->up);
}

void Node::DeleteDown()
{
	free(this->down);
}