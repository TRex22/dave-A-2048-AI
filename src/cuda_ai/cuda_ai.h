/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985
	
	Cuda Ai
*/

#define heading "CUDA Dave Ai for playing 2048 using matrix approach"
#define results_header "children,nodes,time,win rate,flops"
#define results_location "../results/results_cuda1_ai.csv"
#define app_name "Cuda 2048 AI - DaveAi"
    
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stack>

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <curand.h>
#include <curand_kernel.h>
    
/* Global variables */
int board_size = 4;
bool use_rnd = false;
int max_depth = 600;
int max_num_nodes = 6000;
int num_trees = -1;
bool save_to_file = false;
bool print_output = false;
bool print_path = false;
bool save_csv = false;
string initial_state_path = "";
string filepath = "./results/cuda _ai";
bool DEBUG = false;
float time_limit = -1.0;

int num_host_leaves = 1024;//1;//1024; //todo: dynamic calcs
int num_sub_tree_nodes = 1024;//3096;//1024; 

#define DIM 1024
#define warp 32

struct Tree_Stats {
    int BOARD_SIZE = 0;
    Node* root = NULL;
    
    Node* optimal2048 = NULL;
    size_t idx = 0;
    
    int num_nodes = 0;
    int max_depth = 0;   
    int num_solutions = 0;
    int num_leaves = 0;
    int num_cutoff_states = 0;
};

void update_tree_stats(Tree_Stats* stats, Node* root, Node* optimal2048, size_t idx, int num_nodes, int max_depth, int num_solutions, int num_leaves, int num_cutoff_states)
{
    stats->root = root;
    stats->optimal2048 = optimal2048;
    stats->idx = idx;
    stats->num_nodes = num_nodes;
    stats->max_depth = max_depth;
    stats->num_solutions = num_solutions;
    stats->num_leaves = num_leaves;
    stats->num_cutoff_states = num_cutoff_states;
}

/* Function Headers */
int main(int argc, char *argv[]);
void run_AI();
void calc_thread_count(int* threadCount, int height);


/* Serial Tree Build */
std::stack<Node*> get_init_states(int nodes);
int count_computable_nodes(stack<Node*> stack);
bool is_leaf(GameState* state);
void generateChidlren(Node* currentNode, Tree* tree);  

/* device functions */
__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes);
__global__ void build_trees(Node* device_arr, int* device_boards, Tree_Stats* device_tstats, int*** result, int num_sub_tree_nodes, int board_size, curandState_t* rnd_states, size_t height, size_t width, bool print_output, bool DEBUG);
    
/*cuda_2048.cpp*/
__device__ bool cuda_add_new_number(GameState *currentGame, curandState_t* states, int* device_num_sub_tree_nodes);

__device__ void cuda_process_action(GameState *currentGame, int action, int boardSize);
__device__ void cuda_process_left(GameState *currentGame, int boardSize);
__device__ void cuda_process_right(GameState *currentGame, int boardSize);
__device__ void cuda_process_up(GameState *currentGame, int boardSize);
__device__ void cuda_process_down(GameState *currentGame, int boardSize);

__device__ void cuda_print_board(int** currentBoard, int board_size);
  
void process_args(int argc, char *argv[]);
void halt_execution_cuda(string);

void print_cuda_usage(int argc, char *argv[]);

void copy_board(int** to, int** from, int board_size)
{
    for(int i = 0; i < board_size; ++i)
    {
        for (int j = 0; j < board_size; ++j)
        {
            to[i][j] = from[i][j];
        }
    }
}

void print_cuda_usage(int argc, char *argv[])
{   
    printf("At least one parameter must be selected.\n-1 will denote inf value\n");
    printf("num_trees will override max_num_nodes\n\n");
    printf("usage: %s --use_rnd --max_depth=n --max_num_nodes=n --num_trees=n\n", argv[0]);
    printf("\t--save_to_file --print_output --print_path --save_csv\n");
    printf("\t--DEBUG --usage\n");
}

//TODO:CMDLINE Stuff
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
                print_cuda_usage(argc, argv);
                halt_execution_cuda("\nError: board_size must be grater than 1.");
            }
        }
           
        if(contains_string(str, "use_rnd"))
        {
            use_rnd = true;
        }
           
        if(contains_string(str, "max_depth"))
        {
            max_depth = atoi(str.substr(str.find('=') + 1).c_str());
            if(max_depth < 1)
            {
                print_cuda_usage(argc, argv);
                halt_execution_cuda("\nError: max_depth must be grater than 0.");
            }
        }
           
        if(contains_string(str, "max_num_nodes"))
        {
            max_num_nodes = atoi(str.substr(str.find('=') + 1).c_str());
            // num_sub_tree_nodes = max_num_nodes;
            if(max_num_nodes < 2)
            {
                print_cuda_usage(argc, argv);
                halt_execution_cuda("\nError: max_num_nodes must be grater than 1.");
            }
        }
        
        if(contains_string(str, "num_trees"))
        {
            num_trees = atoi(str.substr(str.find('=') + 1).c_str());
            // num_sub_tree_nodes = max_num_nodes;
            if(max_num_nodes < 1)
            {
                print_cuda_usage(argc, argv);
                halt_execution_cuda("\nError: num_trees must be grater than 0.");
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
            print_cuda_usage(argc, argv);
            halt_execution_cuda("");
        }

        if(contains_string(str, "time_limit"))
        {
            time_limit = atof(str.substr(str.find('=') + 1).c_str());
        }
    }
}

void halt_execution_cuda(string message="")
{
    cudaDeviceReset();
	halt_execution(message);
}