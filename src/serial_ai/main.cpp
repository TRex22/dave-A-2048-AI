/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the serial AI approach

	boardSize = since the board is square this is just the width of the board
	score = current player score, starts at 0
	currentBoard = current squares starts at 0 and has all the numbers on the board
	

	AI General Algorithm:
	1. Generate a random init board
	2. loop (in ai using these functions) until 2048
	3. in loop:
		a. guess a move
		b. check new gamestate
		c. retry if fail
		d. go back to a if pass
	4.yay win!
    
    input:
        board size (b)
        lock rnd flag
        max depth (d)
        max nodes (n)
        save to file (f)
        filepath (fp)
        print_cmd (p) t/f
        print_path (path)
        initial state file (is)
        debug (d)
        
    timing
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stack>
#include "omp.h"

#include "../helper/helper.h"

/* Global variables */
#define app_name "Serial 2048 AI - DaveAi"
    
int board_size = 4;
bool use_rnd = false;
int max_depth = -1;
int max_num_nodes = -1;
bool save_to_file = false;
bool print_output = false;
bool print_path = false;
bool save_csv = false;
string initial_state_path = "";
string filepath = "./results/serial_ai";
bool DEBUG = false;

/* Headers */
int main(int argc, char *argv[]);
void run_AI();

void process_args(int argc, char *argv[]);

Tree* buildTree(Tree* tree, int depth_limit, int node_limit);
void generateChidlren(Node* currentNode, Tree* tree);

/* Definitions */
int main(int argc, char *argv[])
{
    print_cmd_heading(app_name);
    if (argc == 1)
    {
        print_usage(argc, argv);
        halt_execution();
    }
    
    if(use_rnd)
        srand(time(NULL));
    else
        srand(10000);

    process_args(argc, argv);
    run_AI();
    
    return EXIT_SUCCESS;
}

void run_AI()
{
    float time_taken = 0.0;
    float start_epoch = omp_get_wtime();
    
    GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);
	buildTree(tree, max_depth, max_num_nodes);
    
    float end_epoch = omp_get_wtime();
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        print_solution(tree);
    }
    
    if(print_output)
    {
        printf("board_size: %i, num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));
        
        if(tree->optimal2048)
            printf("min_depth: %d time_taken: %f\n", tree->optimal2048->depth, time_taken);
    }

    
    if(save_to_file)
    {
        if (save_csv)
            filepath.append(".csv");
        else
            filepath.append(".txt");
                            
        save_solution_to_file(tree, time_taken, filepath, save_csv);
    }
}

Tree* buildTree(Tree* tree, int depth_limit = -1, int node_limit = -1)
{
    //todo: fix this not working correctly
	stack<Node*> tracker;
	tracker.push(tree->root);

	while(!tracker.empty() && !shouldLimit(tree, depth_limit, node_limit))
	{
		Node* currentNode = tracker.top();
        tracker.pop();

		if(currentNode)
		{
			generateChidlren(currentNode, tree);
            
            for (int i = 3; i > -1; --i)
            {
                tracker.push(currentNode->children[i]);
            }
		}
        
        if(DEBUG)
        {
            // printf("%li\n", tracker.size());
        }
	}
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
        // printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        // print_board(currentNode->current_state);
    }
}

void process_args(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        string str = string(argv[i]);
        if(contains_string(str, "board_size"))
        {
            board_size = atoi(str.substr(str.find('=') + 1).c_str());
            if(board_size < 2)
            {
                print_usage(argc, argv);
                halt_execution("\nError: board_size must be grater than 1.");
            }
        }
           
        if(contains_string(str, "use_rnd"))
        {
            use_rnd = true;
        }
           
        if(contains_string(str, "max_depth"))
        {
            max_depth = atoi(str.substr(str.find('=') + 1).c_str());
            if(max_depth < 2)
            {
                print_usage(argc, argv);
                halt_execution("\nError: max_depth must be grater than 1.");
            }
        }
           
        if(contains_string(str, "max_num_nodes"))
        {
            max_num_nodes = atoi(str.substr(str.find('=') + 1).c_str());
            if(max_num_nodes < 2)
            {
                print_usage(argc, argv);
                halt_execution("\nError: max_num_nodes must be grater than 1.");
            }
        }
           
        if(contains_string(str, "save_to_file"))
        {
            save_to_file = true;
        }
           
        if(contains_string(str, "print_output"))
        {
            print_output = true;
        }
           
        if(contains_string(str, "save_csv"))
        {
            save_csv = true;
        }
           
        // if(contains_string(str, "initial_state_path")
        // {
            // initial_state_path = str.substr(str.find("=") + 1)
            //todo: jmc make this and set an initial state
        // }
           
        if(contains_string(str, "filepath"))
        {
            filepath = str.substr(str.find('=') + 1);
        }
                                                                                                           
        if(contains_string(str, "print_path"))
        {
            print_path = true;
        }
           
        if(contains_string(str, "DEBUG"))
        {
            DEBUG = true;
        }
        
        if(contains_string(str, "usage"))
        {
            print_usage(argc, argv);
            halt_execution();
        }
    }
}


