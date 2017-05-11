void save_solution_to_file(Tree* tree, Node* a2048, float time_taken, std::string filename)
{
    string out = "";
    
    double percent_solutions = (double)tree->num_solutions/((double)tree->num_leaves);
    
    ostringstream ss;
     ss << "" << to_string(tree->num_nodes) << " " << to_string(tree->max_depth) << " " << to_string(time_taken) << " " << to_string(tree->num_solutions) << " " << to_string(tree->num_leaves) << " " << to_string(tree->num_cutoff_states) << " " << to_string(percent_solutions) << "\n";
    
    out = ss.str();
    
    out.append(flatten_path(tree, tree->a2048));
    
    write_results_to_file (filename, out);
}

string flatten_path(Tree* tree, Node* final_node)
{
    string out = "";
    out.append(to_string(tree->BOARD_SIZE)).append("\n");
    
    Node* node = final_node;
    while(node && !checkAtRoot(node))
    {
        out += str_node(node->current_state);
        node = node->parent;
    }
    out += str_node(node->current_state);
    
    return out;
}

string str_node(GameState* currentGame)
{
    string out = "";
    int boardSize = currentGame->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			out.append(to_string(currentGame->currentBoard[i][j])).append(" ");
		}
        out.append("\n");
	}
    
    return out;
}

void print_left_most_path(Tree* tree)
{
	Node* node = tree->root;
    while (node)
    { 
        print_board(node->current_state);

        node = node->children[0];
    }
}

void print_right_most_path(Tree* tree)
{
	Node* node = tree->root;
    while (node)
    { 
        print_board(node->current_state);

        node = node->children[3];
    }
}

bool print_solution(Tree* tree)
{
	if(tree->a2048)
    {
    	Node* node = tree->a2048;
    	while(node && !checkAtRoot(node))
    	{
    		print_board(node->current_state);
    		node = node->parent;
    	}
    	print_board(node->current_state);
    }
    else
    	printf("No solution has been found.\n");
    return true;
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

void halt_execution(string message)
{
	cout << message << endl;
    exit(EXIT_FAILURE);
}

void write_results_to_file (std::string filename, std::string results)
{
	ofstream file;
	file.open(filename.c_str(), ios::app);
		file << results;
	file.close();
}

void write_results_to_file (std::string filename, std::string header, std::string results)
{
	ofstream file;
	file.open(filename.c_str(), ios::app);
		file << header << results << endl;
	file.close();
}

void print_cmd_heading(string app_name)
{
	printf("%s\nLiron Mizrahi 708810 \nJason Chalom 711985\n2017\n\n", app_name.c_str());
}

