void save_solution_to_file(Tree* tree, float time_taken, std::string filename, bool csv)
{
    string out = "";
    string delimit = " ";
    
    if(csv)
        delimit = ",";
    
    double percent_solutions = (double)tree->num_solutions/((double)tree->num_leaves);
//     csv:
//     board_size, num_nodes, max_depth, time_taken, num_solutions, num_leaves, num_cutoff_states, percent_solutions
    ostringstream ss;
     ss << to_string(tree->BOARD_SIZE) << delimit << to_string(tree->num_nodes) << delimit << to_string(tree->max_depth) << delimit << to_string(time_taken) << delimit << to_string(tree->num_solutions) << delimit << to_string(tree->num_leaves) << delimit << to_string(tree->num_cutoff_states) << delimit << to_string(percent_solutions);
    
    if(tree->optimal2048)
        ss << to_string(tree->optimal2048->depth);
    ss << "\n";
    
    out = ss.str();
    
    out.append(flatten_path(tree, tree->optimal2048, delimit));
    
    write_results_to_file (filename, out);
}

string flatten_path(Tree* tree, Node* final_node, string delimit)
{
    string out = "";
    
    Node* node = final_node;
    while(node && !checkAtRoot(node))
    {
        out.append(str_node(node->current_state, delimit));
        node = node->parent;
    }
    out += str_node(node->current_state, delimit);
    
    return out;
}

string str_node(GameState* currentGame, string delimit)
{
    string out = "";
    int boardSize = currentGame->boardSize;

	for (int i=0; i < boardSize; i++)
	{
		for (int j=0; j < boardSize; j++)
		{
			out.append(to_string(currentGame->currentBoard[i][j])).append(delimit);
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
	if(tree->optimal2048)
    {
    	Node* node = tree->optimal2048;
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

void halt_execution(string message="")
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

void print_usage(int argc, char *argv[])
{   
    printf("At least one parameter must be selected.\n\n");
    printf("usage: %s --board_size=n --use_rnd --max_depth=n --max_num_nodes=n\n", argv[0]);
    printf("\t--save_to_file --print_output --print_path --save_csv --initial_state_path=p\n");
    printf("\t--filepath=p --DEBUG --usage\n");
}

bool contains_string(string input, string str)
{
    if (input.find(str) != string::npos) {
        return true;
    } 
    return false;
}
