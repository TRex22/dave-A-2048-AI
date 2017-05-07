#include <stack>

#include "../tree/tree.h"

#define DEBUG true

using namespace std;

Tree* buildTree_with_ustack(Tree* tree, int depth_limit, int node_limit);
Tree* buildTree_inplace(Tree* tree, int depth_limit, int node_limit);
Node* get_sibling(Node* node);
void generateChidlren(Node* currentNode, Tree* tree);
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
		Node* currentNode = tracker.top();
        tracker.pop();

		if(!currentNode->isLeaf)
		{
			generateChidlren(currentNode, tree);
            
            for (int i = 0; i < 4; i++)
            {
                tracker.push(currentNode->children[i]);
            }
		}
	}
}

Tree* buildTree_inplace(Tree* tree, int depth_limit = -1, int node_limit = -1)
{
    Node* node = tree->root;
    printf("begin\n");
    generateChidlren(node, tree);
    
    bool condition = true;
    printf("before\n");
    while (condition)
    { 
        printf("in while1\n");
        if(node->children[0])
        {
            printf("first child: %d\n", node->children[0]);
            node = node->children[0];
            printf("node set first child\n");
            generateChidlren(node->children[0], tree); //segfault? why
        }
        else
        {
            printf("no children\n");
            while(!node->hasChildren)
            {
                printf("while no chidlren\n");
                if(checkAtRoot(node))
                    return tree;
                
                Node* nextChild = get_sibling(node);
                if(nextChild != NULL)
                {
                    node = nextChild;
                    generateChidlren(node, tree);
                }
                else
                {
                    node = node->parent;
                }                
            }
        }        
        
        condition = shouldLimit(tree, depth_limit, node_limit);
        printf("condition\n");
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

void generateChidlren(Node* currentNode, Tree* tree)
{
	GameState *state_left;
	GameState *state_right;
	GameState *state_up;
	GameState *state_down;

	GameState* states[] = {state_left, state_right, state_up, state_down};
    printf("init shit\n");
	for (int i = 0; i < 4; i++)
	{
		states[i] = new GameState(tree->BOARD_SIZE);
        printf("board size: %d, statei: %d\n", tree->BOARD_SIZE, currentNode->current_state); // another segfault
		states[i]->copy(currentNode->current_state);
        printf("copied states\n");
		process_action(states[i], i);
		add_new_number(states[i]);
        printf("made state\n");
		int currentDepth = currentNode->depth + 1;
		if(tree->max_depth < currentDepth)
			tree->max_depth = currentDepth;
		
		currentNode->children[i] = new Node(currentNode, states[i], currentDepth);
		tree->num_nodes++;
        printf("set child\n");
        if(!canContinue(currentNode->children[i]) 
           || compare_game_states(currentNode->current_state, currentNode->children[i]->current_state))
        {
            currentNode->children[i]->isLeaf = true;
            printf("is leaf\n");
        }
	}
    
    currentNode->hasChildren = true;
    
    if(DEBUG)
    {
        printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        print_board(currentNode->current_state);
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
            printf("AAAA: %d\n",i);
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