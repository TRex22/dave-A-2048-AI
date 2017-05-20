std::stack<Node*> get_init_states(int nodes)
{
    int computable_nodes = 0;


    GameState* initial_state = new GameState(board_size);
    add_new_number(initial_state);

    Tree* tree = new Tree(initial_state);
    stack<Node*> tracker;

    tracker.push(tree->root);

    do
    {
        Node* currentNode = tracker.top();
        tracker.pop();

        if(currentNode)
        {
            generateChidlren(currentNode, tree);
            
            for (int i = 3; i > -1; --i)
            {
                if (currentNode->children[i])
                {
                    tracker.push(currentNode->children[i]);
                }
                
            }
        }
        computable_nodes = count_computable_nodes(tracker);

    }while(computable_nodes < nodes);
    
    if(print_output)
        printf("DONE GETTING %d INIT STATES\n", count_computable_nodes(tracker));

    return tracker;
}

int count_computable_nodes(stack<Node*> stack)
{
    int count = 0;
    std::stack<Node*> tracker;

    while(!stack.empty())
    {
        Node* node = stack.top();
        if( !determine_2048(node->current_state) && !is_leaf(node->current_state) )
        {
            count++;
        }
        stack.pop();
        tracker.push(node);
    }

    while(!tracker.empty())
    {
        Node* node = tracker.top();
        tracker.pop();
        stack.push(node);
    }

    return count;
}

bool is_leaf(GameState* state)
{
    bool result = true;
    for (int i = 0; i < 4; i++)
    {
        GameState* newState = new GameState(board_size);
        newState->copy(state);
        process_action(newState, i);

        if(compare_game_states(state, newState) == false)
        {
            result = false;
            return result;
        }
    }
    return result;
}

void generateChidlren(Node* currentNode, Tree* tree)
{
	for (int i = 0; i < 4; i++)
	{
        GameState* newState = new GameState(tree->BOARD_SIZE);
		newState->copy(currentNode->current_state);

		process_action(newState, i);

        if(!determine_2048(currentNode->current_state) && !compare_game_states(currentNode->current_state, newState))
        {
            bool fullBoard = !add_new_number(newState);
            if(!fullBoard)
            {
                int currentDepth = currentNode->depth + 1;
                if(tree->max_depth < currentDepth)
                    tree->max_depth = currentDepth;
                
                currentNode->children[i] = new Node(currentNode, newState, currentDepth);
                tree->num_nodes++;

                currentNode->hasChildren = true;
            }
            else
            {
                currentNode->children[i] = nullptr;
            }
        }
        else
        {
            currentNode->children[i] = nullptr;
        }

        if(determine_2048(currentNode->current_state)) //win and shortest path
        {
            if(tree->optimal2048)
            {
                if(currentNode->depth < tree->optimal2048->depth) 
                    tree->optimal2048 = currentNode;
            }
            else
                tree->optimal2048 = currentNode;
            
            tree->num_solutions++;
        }

        if(determine_2048(currentNode->current_state) || compare_game_states(currentNode->current_state, newState)) //leaf
        {
            tree->num_leaves++;
        }
	}
    
    if(DEBUG)
    {
        printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        print_board(currentNode->current_state);
    }
}