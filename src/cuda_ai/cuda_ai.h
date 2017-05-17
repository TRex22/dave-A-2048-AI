/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985
	
	Cuda Ai
*/

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
__global__ void buildTree(Node* device_arr, Tree_Stats* device_tstats, int num_sub_tree_nodes, int board_size, curandState_t* rnd_states, size_t height, size_t width, size_t nodeArrSize);
__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes);

/*cuda_2048.cpp*/
__device__ bool cuda_add_new_number(GameState *currentGame, curandState_t* states, int* device_num_sub_tree_nodes);

__device__ void cuda_process_action(GameState *currentGame, int action, int boardSize);
__device__ void cuda_process_left(GameState *currentGame, int boardSize);
__device__ void cuda_process_right(GameState *currentGame, int boardSize);
__device__ void cuda_process_up(GameState *currentGame, int boardSize);
__device__ void cuda_process_down(GameState *currentGame, int boardSize);
  
void process_args(int argc, char *argv[]);
void halt_execution_cuda(string);