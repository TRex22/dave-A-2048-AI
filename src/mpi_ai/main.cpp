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
#include "mpi.h"

#include "../helper/helper.h"

/* Global variables */
#define app_name "MPI 2048 Parallel AI - DaveAi"
    
int board_size = 4;
bool use_rnd = false;
int max_depth = -1;
int max_num_nodes = -1;
bool save_to_file = false;
bool print_output = false;
bool print_path = false;
bool save_csv = false;
string initial_state_path = "";
string filepath = "./results/mpi_ai";
bool DEBUG = false;
float time_limit = -1.0;

/* Headers */
int main(int argc, char *argv[]);
void run_AI(GameState* state);

void process_args(int argc, char *argv[]);

Tree* buildTree(Tree* tree, int depth_limit, int node_limit, float start_time, float time_limit);
void generateChidlren(Node* currentNode, Tree* tree);

int log_4(int comm_sz);
std::stack<Node*> get_init_states(int nodes);
void linearize_and_send(std::stack<Node*> stack, int comm_sz);
int count_computable_nodes(stack<Node*> stack);
bool is_leaf(GameState* state);
Node* get_optimal_leaf(std::stack<Node*> init_states, int optimal_proc);
void save_subtree_to_file(std::stack<Node* >init_states);
void save_subtree_to_file(std::stack<Node*> states, int boardSize);
std::stack<Node*> push_parents_to_stack(Node* leaf);
std::stack<Node*> push_parents_to_stack(Node* leaf,int min_depth);
void save_and_send_stack_as_matrix(std::stack<Node*> states);


/* Definitions */
int main(int argc, char *argv[])
{
    int myrank, comm_sz;
    int local_size = 0;
    int total_num_nodes = 0;
    int total_max_depth = 0;
    int total_sols = 0;
    int toal_leaves = 0;
    
    //start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Status status;

    if (myrank == 0)
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

        std::stack<Node*> init_states;
        init_states = get_init_states(comm_sz);
        linearize_and_send(init_states, comm_sz);

        int local_optima[comm_sz-1];
        int optimum;

        //Recv local optima from each process and store it in process_rank-1 index
        for (int i = 1; i < comm_sz; ++i)
        {
            MPI_Recv(&optimum, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            local_optima[src - 1] = optimum;
        }

        //find min solution path and its process
        int min = local_optima[0];
        int optimal_proc = 0;
        for (int i = 1; i < comm_sz-1; ++i)
        {
            if (local_optima[i] != -1)
            {
                if (local_optima[i] < min)
                {
                    min = local_optima[i];
                    optimal_proc = i;
                }
            }
        }

        Node* optimal_leaf = get_optimal_leaf(init_states, optimal_proc + 1);



        //push path from optimal leaf to root into stack optimal_subtree
        std::stack<Node*> optimal_subtree;
        Node* node = optimal_leaf;
        optimal_subtree.push(node);

        while(node->parent != NULL)
        {
            node = node->parent;
            optimal_subtree.push(node);
        }




        //If there is a solution
        if (min != -1)
        {
			//prints optimal subtree
        	save_subtree_to_file(optimal_subtree, board_size);

            
            for (int i = 0; i < comm_sz-1; ++i)
            {
                printf("%d ", local_optima[i]);
            }
            printf("\noptimal_proc: %d\n", optimal_proc+1);
            printf("min: %d\n", min);
            

            //Send a '1' to proc with best solution
            int one = 1;
            int zero = 0;
            for (int i = 0; i < comm_sz-1; ++i)
            {
                if (i != optimal_proc)
                {
                    MPI_Send(&zero, 1, MPI_INT, i+1, i+1, MPI_COMM_WORLD);
                }
                else if (i == optimal_proc)
                {
                    MPI_Send(&one, 1, MPI_INT, i+1, i+1, MPI_COMM_WORLD);
                }
            }
        }
        else if (min == -1)
        {
            printf("NO SOLUTION!!!!!!\n");
            //Send a '0' to all procs
            int zero = 0;
            for (int i = 0; i < comm_sz-1; ++i)
            {
                MPI_Send(&zero, 1, MPI_INT, i+1, i+1, MPI_COMM_WORLD);
            }
        }
        
        

    }

    //Else if not Proc 0
    else
    {
        MPI_Recv(&local_size, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD, &status);
        int square_size = local_size*local_size;
        int local_init_board[ square_size ];
        MPI_Recv(local_init_board, square_size, MPI_INT, 0, myrank, MPI_COMM_WORLD, &status);
        MPI_Recv(&max_num_nodes, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD, &status);
        printf("Proc %d received data.\n", myrank);

        GameState* init_state = new GameState(local_size);
        for (int i = 0; i < local_size; ++i)
        {
            for (int j = 0; j < local_size; ++j)
            {
                init_state->currentBoard[i][j] = local_init_board[i*local_size + j];
            }
        }
        run_AI(init_state);
    }



    MPI_Finalize();
    return EXIT_SUCCESS;
}


-

void linearize_and_send(std::stack<Node*> stack, int comm_sz)
{
    std::stack<Node*> stack2;
    int size = board_size;

    for (int k = 1; k < comm_sz; ++k)
    {
        MPI_Request request;        
        int board[size*size];

        Node* node = stack.top();
        stack2.push(node);
        stack.pop();
        node->process_num = k;

        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                board[i*size + j] = node->current_state->currentBoard[i][j];
            }
        }

        MPI_Send(&size, 1, MPI_INT, k, k, MPI_COMM_WORLD);
        MPI_Send(board, size*size, MPI_INT, k, k, MPI_COMM_WORLD);
        MPI_Send(&max_num_nodes, 1, MPI_INT, k, k, MPI_COMM_WORLD);
        printf("sent init state to proc: %d\n", k);
    }

    while(!stack2.empty())
    {
        Node* top = stack2.top();
        stack.push(top);
        stack2.pop();
    }
}

void run_AI(GameState* state)
{
    printf("running AI\n");
    float time_taken = 0.0;
    int myrank;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    float start_epoch = omp_get_wtime();
    
    GameState* initial_state = state;

	Tree* tree = new Tree(initial_state);
	buildTree(tree, max_depth, max_num_nodes, start_epoch, time_limit);
    
    float end_epoch = omp_get_wtime();
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        print_solution(tree);
    }
    
    // if(print_output)
    // {
        // printf("board_size: %i, num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));
        
        // if(tree->optimal2048)
        //     printf("min_depth: %d time_taken: %f\n", tree->optimal2048->depth, time_taken);
    // }

    int optimum;
    if(tree->optimal2048)
        optimum = tree->optimal2048->depth;
    else
        optimum = -1;

    MPI_Send(&optimum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    int is_best_proc;
    MPI_Recv(&is_best_proc, 1, MPI_INT, 0, myrank, MPI_COMM_WORLD, &status);

    if (is_best_proc == 1)
    {
        printf("Proc %d: I am the best proc :D\n", myrank);
        std::stack<Node*> optimal_solution = push_parents_to_stack(tree->optimal2048, tree->optimal2048->depth);
        save_subtree_to_file(optimal_solution);
    }
    else if (is_best_proc == 0)
    {
        printf("Proc %d: I am not the best proc ;_;\n", myrank);
    }


    // if(save_to_file)
    // {
    //     if (save_csv)
    //         filepath.append(".csv");
    //     else
    //         filepath.append(".txt");
                            
    //     save_solution_to_file(tree, time_taken, filepath, save_csv);
    // }
}

Tree* buildTree(Tree* tree, int depth_limit = -1, int node_limit = -1, float start_time = -1.0, float time_limit=-1.0)
{
    //todo: fix this not working correctly
	stack<Node*> tracker;
	tracker.push(tree->root);

    float currentTime = omp_get_wtime()-start_time;

	while(!tracker.empty() && !shouldLimit(tree, depth_limit, node_limit, currentTime, time_limit))
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
            printf("%lui\n", tracker.size());
        }

	    currentTime += omp_get_wtime()-start_time;
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
        printf("%d, %d\n", tree->num_nodes, tree->max_depth);
        print_board(currentNode->current_state);
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

        if(contains_string(str, "time_limit"))
        {
            time_limit = atof(str.substr(str.find('=') + 1).c_str());
        }
    }
}

int log_4(int comm_sz)
{
    /* calculates log_4(comm_sz) from log_10(comm_sz)/log_10(4) */
    int a = log(comm_sz);
    int b = log(4);
    int c = a/b;
    return c;
}

Node* get_optimal_leaf(std::stack<Node*> init_states, int optimal_proc)
{
    while(!init_states.empty())
    {
        Node* node = init_states.top();
        if (node->process_num == optimal_proc)
        {
            printf("Proc 0: found opt Node at proc - %d\n", node->process_num);
            return node;
        }
        init_states.pop();
    }
}

std::stack<Node*> push_parents_to_stack(Node* leaf)
{
    std::stack<Node*> stack;
    Node* node = leaf;
    stack.push(node);

    while(node->parent != NULL)
    {
        node = node->parent;
        stack.push(node);
    }

    return stack;
}

std::stack<Node*> push_parents_to_stack(Node* leaf, int min_depth)
{
    std::stack<Node*> stack;
    Node* node = leaf;
    stack.push(node);

    for (int i = 0; i < min_depth; ++i)
    {
        node = node->parent;
        stack.push(node);
    }

    return stack;
}

void save_subtree_to_file(std::stack<Node*> states)
{
    Node* node = states.top();
    int size = node->current_state->boardSize;
    printf("got board size %d\n", size);

    while(!states.empty())
    {
        node = states.top();
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                printf("%d", node->current_state->currentBoard[i][j]);
                if (i != size-1 || j != size-1)
                {
                    printf(",");
                }
            }
        }
        printf("\n");
        states.pop();
    }
}

void save_subtree_to_file(std::stack<Node*> states, int boardSize)
{
    states.pop();
    states.pop();
    Node* node = states.top();
    int size = boardSize;
    printf("got board size %d\n", size);

    while(!states.empty())
    {
        printf("!states.empty()\n");
        node = states.top();
        printf("node = top\n");
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                printf("%d", node->current_state->currentBoard[i][j]);
                
                if (i != size-1 || j != size-1)
                {
                    printf(",");
                }
            }
        }
        printf("\n");
        states.pop();
    }
}

void save_and_send_stack_as_matrix(std::stack<Node*> states)
{
    Node* node = states.top();
    int size = node->current_state->boardSize;

    // printf("...rows = %d\n", max_state_size);

    // for (int k = 0; k < max_state_size; ++k)
    // {
    //     for (int i = 0; i < size; ++i)
    //     {
    //         for (int j = 0; j < size; ++j)
    //         {
    //             state_mat[k][i*size + j] = node->current_state->currentBoard[i][j];
    //         }
    //     }
    //     node = states.top();
    // }
    // states.pop();

    // printf("...done matrix\n");

    // //send matrix size and the actual matrix
    // MPI_Send(&max_state_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    // printf("...sent size\n");
    // MPI_Send(&state_mat, max_state_size * size*size, MPI_INT, 0, 0, MPI_COMM_WORLD);

    // free(state_mat[0]);
    // free(state_mat);

    while(!states.empty())
    {
        node = states.top();
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < size; ++j)
            {
                printf("%d", node->current_state->currentBoard[i][j]);
                if (i != size-1 && j != size-1)
                {
                    printf(",");
                }
            }
            printf("\n");
        }
        states.pop();
    }
}
