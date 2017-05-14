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
dim3 dimBlock( warp, warp, 1 );
dim3 dimGrid( board_size*warp, board_size*warp );

// Auto-Verification Code
bool testResult = true;

using namespace std;

/* Function Headers */
int main(int argc, char *argv[]);

/* device functions */
__global__ void buildTree(Tree* tree, stack<Node*> tracker, int depth_limit, int node_limit, float start_time, float time_limit);
__device__ void generateChidlren(Node* currentNode, Tree* tree);

/* host functions */
void run_AI();
char** initTreeArray(int num_nodes, int num_children);
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
    // char ** hostTreeArray = initTreeArray(MAX_NODES, MAX_CHILDREN);
    float time_taken = 0.0;
      
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);    
    float start_epoch = sdkGetTimerValue(&timer);
    
    GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);
    stack<Node*> tracker;
    checkCudaErrors(cudaMalloc((void **) &tracker, sizeof(Node)));
    
	buildTree<<<dimGrid, dimBlock>>>(tree, tracker, max_depth, max_num_nodes, start_epoch, time_limit);
    
    float end_epoch = sdkGetTimerValue(&timer);
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
    
    /* cleanup */
    sdkDeleteTimer(&timer);
	
    /*cuda vars*/
	/*float* imageData = NULL;
	float* kernelData = NULL;
	float* resultArr = NULL;

	checkCudaErrors(cudaMalloc((void **) &imageData, image.size));
	checkCudaErrors(cudaMalloc((void **) &kernelData, kernel.size));
	checkCudaErrors(cudaMalloc((void **) &resultArr, result.size));

	checkCudaErrors(cudaMemcpy(imageData, image.hData, image.size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(kernelData, kernel.hData, kernel.size, cudaMemcpyHostToDevice));

	dim3 dimBlock( 32, 32, 1 );
	dim3 dimGrid( image.height/32, image.width/32 );

	int borderPadding = get_boundary_size(kernel);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	naive_parallel_convolution_kernel<<<dimGrid, dimBlock>>>(imageData, image.width, image.height, 
		kernelData, kernel.width, kernel.height, resultArr, borderPadding);

	sdkStopTimer(&timer);
    printf("\nProcessing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (image.dim / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

    checkCudaErrors(cudaMemcpy(result.hData, resultArr, result.size, cudaMemcpyDeviceToHost));
    cudaFree(imageData);
    cudaFree(kernelData);
    cudaFree(resultArr);

    save_pgm_image(result, result_image_location);*/
}

__global__ void buildTree(Tree* tree, stack<Node*> tracker, int depth_limit = -1, int node_limit = -1, float start_time = -1.0, float time_limit=-1.0)
{
    //todo: fix this not working correctly
    
	tracker.push(tree->root);

    // StopWatchInterface *timer = NULL;
    // sdkCreateTimer(&timer);
    // sdkStartTimer(&timer);    
    // float currentTime = sdkGetTimerValue(&timer)-start_time;
    
	while(!tracker.empty() && !shouldLimit(tree, depth_limit, node_limit, -1.0, -1.0))
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
        
        // if(DEBUG)
        // {
        //     printf("%lui\n", tracker.size());
        // }

	    // currentTime += sdkGetTimerValue(&timer)-start_time;
    }
    
    // sdkDeleteTimer(&timer);
}

// char* initTreeArray(int num_nodes, int num_children)
// {
// 	char *treeArr = (char *) malloc(max_num_nodes);
    
//     return treeArr;
// }

__device__ void generateChidlren(Node* currentNode, Tree* tree)
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
    
    // if(DEBUG)
    // {
    //     printf("%d, %d\n", tree->num_nodes, tree->max_depth);
    //     print_board(currentNode->current_state);
    // }
}

/*__global__ void naive_parallel_convolution_kernel(float* imageData, int imageWidth, int imageHeight, float* kernelData, int kernelWidth, int kernelHeight, float* resultArr, int borderPadding)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y*imageWidth+x;

    int kernel_woffset = kernelWidth/2;
    int kernel_hoffset = kernelHeight/2;

    //check if not border
    if ((x > borderPadding && y > borderPadding) && (x < imageWidth - borderPadding && y < imageHeight - borderPadding))
    {
    	float sum = 0.0;
    	int count = 0;

    	for (int k = -kernel_hoffset; k <= kernel_hoffset; k++)
    	{
    		for (int l = -kernel_woffset; l <= kernel_woffset; l++)
			{
				sum += kernelData[count] * imageData[idx + k*imageWidth + l];
				count++;
			}
    	}

		if(sum < 0.0) sum = 0.0;
		else if (sum > 1.0) sum = 1.0;

		resultArr[idx] = sum;
    }
}*/

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
	halt_execution(message);
	cudaDeviceReset();
}