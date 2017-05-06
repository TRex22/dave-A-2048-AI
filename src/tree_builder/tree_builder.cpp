#include <stack>

#include "../tree/tree.h"

using namespace std;

Tree* buildTree_with_ustack(Tree* tree, int depth_limit, int node_limit);
Tree* buildTree_inplace(Tree* tree);
Node* get_sibling(Node* node);
void generateChidlren(Node* currentNode, Tree* tree, stack<Node*> tracker, int board_size);
bool checkAtRoot(Node* node);
bool canContinue(Node* node);
bool shouldLimit(Tree* tree, int depth_limit, int node_limit);

Tree* buildTree_with_ustack(Tree* tree, int depth_limit = -1, int node_limit = -1)
{
    //todo: fix this not working correctly
	stack<Node*> tracker;
	tracker.push(tree->root);

	while(!tracker.empty() && !shouldLimit(tree, depth_limit, node_limit))
	{
        printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        print_board(tracker.top()->current_state);
		int board_size = 4; //TODO: make as parameter to do this properly
		Node* currentNode = tracker.top();
        tracker.pop();

		if(!currentNode->isLeaf)
		{
			generateChidlren(currentNode, tree, tracker, board_size);
            
            for (int i = 0; i < 4; i++)
            {
                tracker.push(currentNode->children[i]);
            }
		}
	}
}

Tree* buildTree_inplace(Tree* tree)
{
    Node* node = tree->root;
    bool condition = true;
    while (condition)
    {
        if(!node->children[0]->isLeaf)
        {
            node = node->children[0];
        }
        else
        {
            while(!node->hasChildren)
            {
                if(checkAtRoot(node))
                    return tree;
                
                Node* nextChild = get_sibling(node);
                if(nextChild != NULL)
                {
                    node = nextChild;
                }
                else
                {
                    node = node->parent;
                }                
            }
        }
    }
    
    // visit(node)
    // if node.first_child:
    //     node = node.first_child      # walk down
    // else:
    //     while not node.next_sibling:
    //         if node is root:
    //             return
    //         node = node.parent       # walk up ...
    //     node = node.next_sibling     # ... and right
                    
}

void generateChidlren(Node* currentNode, Tree* tree, stack<Node*> tracker, int board_size)
{
	GameState *state_left;
	GameState *state_right;
	GameState *state_up;
	GameState *state_down;

	GameState* states[] = {state_left, state_right, state_up, state_down};

	for (int i = 3; i >= 0; --i)
	{
		states[i] = new GameState(board_size);
		states[i]->copy(currentNode->current_state);
		process_action(states[i], i);
		add_new_number(states[i]);

		int currentDepth = currentNode->depth + 1;
		if(tree->max_depth < currentDepth)
			tree->max_depth = currentDepth;
		
		currentNode->children[i] = new Node(currentNode, states[i], currentDepth);
		tree->num_nodes++;
        
        if(!canContinue(currentNode->children[i]))
        {
            currentNode->children[i]->isLeaf = true;
        }
	}	
}

Node* get_sibling(Node* node)
{
    //return 0 if finished
    Node* parent = node->parent;
    
    // easier to do by hand not to use <algorithm>
    int idex = 0;
    for(int i = 0; i < 4; ++i)
    {
        if(&parent->children[i] == &node) // Are the memory addresses the same
        {
            idex = i;
        }
    }
    
    if(idex > 0 && idex != 3) //cant be last index
    {
        return parent->children[idex+1];
    }

    return NULL;
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

	return false;
}

bool shouldLimit(Tree* tree, int depth_limit, int node_limit)
{
    // stop is true or false
    if(tree->max_depth > depth_limit-1 && depth_limit != -1)
    {
        return true;
    }
    
    if(tree->num_nodes > node_limit-1 && node_limit != -1)
    {
        return true;
    }
    
    return false;
}