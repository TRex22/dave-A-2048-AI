/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985
	
	Cuda Ai
*/

#define CUDA True //this is to use the same library functions

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stack>

#include "../helper/helper.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <curand.h>
#include <curand_kernel.h>

#define heading "CUDA Dave Ai for playing 2048 using matrix approach"
#define results_header "children,nodes,time,win rate,flops"
#define results_location "../results/results_cuda1_ai.csv"

/* Global variables */
#define app_name "Cuda 2048 AI - DaveAi"
    
int board_size = 4;
bool use_rnd = false;
int max_depth = -1;
int max_num_nodes = 10000;
bool save_to_file = false;
bool print_output = false;
bool print_path = false;
bool save_csv = false;
string initial_state_path = "";
string filepath = "./results/cuda _ai";
bool DEBUG = false;
float time_limit = -1.0;

#define DIM 1024
#define warp 32

// Auto-Verification Code
bool testResult = true;

using namespace std;

/* Function Headers */
int main(int argc, char *argv[]);

/* device functions */
__global__ void buildTree(Node* device_arr, int* device_num_sub_tree_nodes, int* board_size);
__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes);
__device__ bool cuda_add_new_number(GameState *currentGame, curandState_t* states, int* device_num_sub_tree_nodes);

/* host functions */
void run_AI();
void serialBuildTree(Tree* tree, int leaf_node_limit, Node* host_arr);
void serialGenerateChidlren(Node* currentNode, Tree* tree, Node* host_arr);
Node* createHostTreeArray(Tree* tree, int num_host_leaves, int num_sub_tree_nodes);
    
void process_args(int argc, char *argv[]);
void halt_execution_cuda(string);

int main(int argc, char *argv[])
{
	//some basic setup
    print_cmd_heading(app_name);
    
    if (argc == 1)
    {
        print_usage(argc, argv);
        halt_execution_cuda("");
    }
    
    if(use_rnd)
        srand(time(NULL));
    else
        srand(10000);
    
    process_args(argc, argv);
    run_AI();

	cudaDeviceReset();
    printf("%s completed, returned %s\n",
           heading,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void run_AI()
{
    if(print_output)
        printf("Building initial tree...\n");
    
    float time_taken = 0.0;
    
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);    
    float start_epoch = sdkGetTimerValue(&timer);
    
    GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);
    stack<Node*> tracker;
    
    int num_host_leaves = 1024; //todo: dynamic calcs
    int num_sub_tree_nodes = 1024;
    int nodeArrSize = (num_host_leaves+4)*num_sub_tree_nodes*sizeof(Node);
    Node* host_arr = (Node*)malloc(nodeArrSize);
    
    serialBuildTree(tree, num_host_leaves, host_arr);
    
    if(print_output)
        printf("Move host array to device...\n");
    
    // device variables
    Node* device_arr;
    int* device_num_sub_tree_nodes;
    int* device_board_size;
    
    checkCudaErrors(cudaMalloc((void**)&device_arr, nodeArrSize));
    checkCudaErrors(cudaMalloc((void**)&device_num_sub_tree_nodes, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&device_board_size, sizeof(int)));
    checkCudaErrors(cudaMemcpy(device_arr, host_arr, nodeArrSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_num_sub_tree_nodes, &max_num_nodes, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_board_size, &board_size, sizeof(int), cudaMemcpyHostToDevice));
    
    dim3 dimBlock( warp, warp, 1 );
    dim3 dimGrid( board_size*warp, board_size*warp );
    
    if(print_output)
        printf("Start buildTree kernel...\n");
    
    //curand stuff
    unsigned long int seed;
    if(use_rnd)
        seed = time(NULL);
    else
        seed = 10000;
    
    curandState_t* states;
    cudaMalloc((void**) &states, warp*warp* sizeof(curandState_t)); //N*sizeof
    
    init_rnd<<<dimGrid, dimBlock>>>(seed, states, device_num_sub_tree_nodes);
	buildTree<<<dimGrid, dimBlock>>>(device_arr, device_num_sub_tree_nodes, device_board_size);
    
    if(print_output)
        printf("Copy results back to host...\n\n");
    cudaMemcpy(host_arr, device_arr, nodeArrSize, cudaMemcpyDeviceToHost);
    
    float end_epoch = sdkGetTimerValue(&timer);
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        // print_solution(tree);
    }
    
    if(print_output)
    {
//         printf("board_size: %i, num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, ((double)tree->num_solutions/(double)tree->num_leaves));
        
//         if(tree->optimal2048)
//             printf("min_depth: %d time_taken: %f\n", tree->optimal2048->depth, time_taken);
    }

    
    if(save_to_file)
    {
        printf("Save optimal path to file...\n");
//         if (save_csv)
//             filepath.append(".csv");
//         else
//             filepath.append(".txt");
                            
//         save_solution_to_file(tree, time_taken, filepath, save_csv);
    }
    
    /* cleanup */
    sdkDeleteTimer(&timer);
	checkCudaErrors(cudaFree(device_arr));
    checkCudaErrors(cudaFree(device_num_sub_tree_nodes));
    checkCudaErrors(cudaFree(device_board_size));
}

__global__ void buildTree(Node* device_arr, int* device_num_sub_tree_nodes, int* board_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = (*device_num_sub_tree_nodes)*y+x;
    
    int curr_node = 0;
    
    while(curr_node < *device_num_sub_tree_nodes-4)
    {
        Node* currentNode = &device_arr[x][curr_node];
        for (int i = 0; i < 4; i++)
        {
            GameState* newState = new GameState(*board_size);
            newState->copy(currentNode->current_state);

            process_action(newState, i);

            if(!determine_2048(currentNode->current_state) && !compare_game_states(currentNode->current_state, newState))
            {
                bool fullBoard = !add_new_number(newState);
                if(!fullBoard)
                {
                    int currentDepth = currentNode->depth + 1;
                    // if(tree->max_depth < currentDepth)
                    //     tree->max_depth = currentDepth;

                    currentNode->children[i] = new Node(currentNode, newState, currentDepth);
                    // tree->num_nodes++;

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
        }
        
        curr_node++;
    }
}

__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = (*device_num_sub_tree_nodes)*y+x;
    curand_init(seed, idx, 0, &states[blockIdx.x]);
}

__device__ bool cuda_add_new_number(GameState *currentGame, curandState_t* states, int* device_num_sub_tree_nodes)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = (*device_num_sub_tree_nodes)*y+x;
    
    int rand_row = curand(&states[idx]) % currentGame->boardSize;
    int rand_col = curand(&states[idx]) % currentGame->boardSize;

	if(checkBoardEmptySlot(currentGame))
	{
		while(currentGame->currentBoard[rand_row][rand_col] != 0)
		{
            rand_row = curand(&states[idx]) % currentGame->boardSize;
            rand_col = curand(&states[idx]) % currentGame->boardSize;
		}

		currentGame->currentBoard[rand_row][rand_col] = 2;
		return true;
	}
	return false;
}

void serialBuildTree(Tree* tree, int leaf_node_limit, Node* host_arr)
{
    //todo: fix this not working correctly
	stack<Node*> tracker;
	tracker.push(tree->root);

	while(!tracker.empty() && !shouldLimit(tree, leaf_node_limit))
	{
		Node* currentNode = tracker.top();
        tracker.pop();
    
		if(currentNode)
		{
			serialGenerateChidlren(currentNode, tree, host_arr);
            
            for (int i = 3; i > -1; --i)
            {
                tracker.push(currentNode->children[i]);
            }
		}
        
        if(DEBUG)
        {
            printf("%lui\n", tracker.size());
        }
    }
}

void serialGenerateChidlren(Node* currentNode, Tree* tree, Node* host_arr)
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

        if(determine_2048(currentNode->current_state) || compare_game_states(currentNode->current_state, newState)) 
        {
            tree->num_leaves++;
        }
        
        if(!determine_2048(currentNode->current_state) && !compare_game_states(currentNode->current_state, newState)) 
        {
            host_arr[tree->num_cutoff_states] = *currentNode;
            tree->num_cutoff_states++;
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
            if(max_depth < 2)
            {
                print_usage(argc, argv);
                halt_execution_cuda("\nError: max_depth must be grater than 1.");
            }
        }
           
        if(contains_string(str, "max_num_nodes"))
        {
            max_num_nodes = atoi(str.substr(str.find('=') + 1).c_str());
            if(max_num_nodes < 2)
            {
                print_usage(argc, argv);
                halt_execution_cuda("\nError: max_num_nodes must be grater than 1.");
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
            print_usage(argc, argv);
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