/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985
	
	Cuda Ai
*/

#define CUDA True //this is to use the same library functions
    
#include "../helper/helper.h"
#include "cuda_ai.h"
#include "cuda_2048.cpp"
#include "serial_tree_builder.cpp"

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

int num_host_leaves = 32;//1024; //todo: dynamic calcs
int num_sub_tree_nodes = 1024; 

#define DIM 1024
#define warp 32

// Auto-Verification Code
bool testResult = true;

using namespace std;

int main(int argc, char *argv[])
{
	//some basic setup
    print_cmd_heading(app_name);
    
    if (argc == 1)
    {
        print_usage(argc, argv);
        halt_execution_cuda("");
    }

    process_args(argc, argv);
    
    if(use_rnd)
        srand(time(NULL));
    else
        srand(10000);
    

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
        printf("Init...\n");
    
    float time_taken = 0.0;
    
    Tree_Stats *tstats = new Tree_Stats;
    tstats->BOARD_SIZE = board_size;
    
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);    
    float start_epoch = sdkGetTimerValue(&timer);
    
    GameState* initial_state = new GameState(board_size);
	add_new_number(initial_state);

	Tree* tree = new Tree(initial_state);
    stack<Node*> tracker;
      
    size_t height = num_host_leaves; 
    size_t width = (num_sub_tree_nodes);
    size_t nodeArrSize = height*width *sizeof(Node);
                                                                                                                              
    if(print_output)
        printf("Allocate host arr...\n");
    Node* host_arr = (Node*)malloc(nodeArrSize);
    
    if(print_output)
        printf("Building initial tree...\n");

    std::stack<Node*> init_states;
    init_states = get_init_states(num_host_leaves); //gets all the cut off nodes for gpu
    
    for(unsigned int i = 0;i < height;i++)
    {
        host_arr[i*width] = *init_states.top();
        init_states.pop();
    }
    
    //update tree stats
    update_tree_stats(tstats, tree->root, tree->optimal2048, 0, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, tree->num_cutoff_states);
    
    if(print_output)
        printf("Move host array to device...\n");
    
    // device variables
    Node* device_arr;
    Tree_Stats* device_tstats;
    int* device_num_sub_tree_nodes;
    
    int threadCounts[2] = {0, 0};
    calc_thread_count(threadCounts, height);
    dim3 dimBlock( threadCounts[0], threadCounts[1], 1 );
	dim3 dimGrid( 1, 1 );
    
    checkCudaErrors(cudaMalloc((void**)&device_arr, nodeArrSize));
    checkCudaErrors(cudaMalloc((void**)&device_tstats, sizeof(Tree_Stats)));
    checkCudaErrors(cudaMalloc((void**)&device_num_sub_tree_nodes, sizeof(int)));
    checkCudaErrors(cudaMemcpy(device_arr, host_arr, nodeArrSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_tstats, tstats, sizeof(Tree_Stats), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_num_sub_tree_nodes, &max_num_nodes, sizeof(int), cudaMemcpyHostToDevice));
    
    if(print_output)
        printf("Start buildTree kernel...\n");
    
    //curand stuff
    unsigned long int seed;
    if(use_rnd)
        seed = time(NULL);
    else
        seed = 10000;
    
    curandState_t* states;
    cudaMalloc((void**) &states, threadCounts[0]*threadCounts[1]*sizeof(curandState_t));
    
    init_rnd<<<dimGrid, dimBlock>>>(seed, states, device_num_sub_tree_nodes);
	buildTree<<<dimGrid, dimBlock>>>(device_arr, device_tstats, num_sub_tree_nodes, board_size, states, height, width, nodeArrSize);
    
    if(print_output)
        printf("Copy results back to host...\n\n");
    
    checkCudaErrors(cudaMemcpy(host_arr, device_arr, nodeArrSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tstats, device_tstats, sizeof(Tree_Stats), cudaMemcpyDeviceToHost));
    
    float end_epoch = sdkGetTimerValue(&timer);
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        // print_solution(tree);
    }
    
    if(print_output)
    {
        printf("board_size: %i, num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, tstats->num_nodes, tstats->max_depth, tstats->num_solutions, tstats->num_leaves, ((double)tstats->num_solutions/(double)tstats->num_leaves));
        
        if(tstats->optimal2048)
            printf("min_depth: %d time_taken: %f\n", tstats->optimal2048->depth, time_taken);
    }

    
    if(save_to_file)
    {
        printf("Save optimal path to file...\n");
        if (save_csv)
            filepath.append(".csv");
        else
            filepath.append(".txt");
                            
        save_solution_to_file(tree, time_taken, filepath, save_csv);
    }
    
    /* cleanup */
    sdkDeleteTimer(&timer);
	checkCudaErrors(cudaFree(device_arr));
    checkCudaErrors(cudaFree(device_tstats));
    checkCudaErrors(cudaFree(device_num_sub_tree_nodes));
}

void calc_thread_count(int* threadCount, int height)
{
    if (height <= DIM)
    {
        threadCount[0] = height;
        threadCount[1] = 1;
    }
    else
    {
        double check = height / DIM;
        check = ceil(check);
        
        threadCount[0] = DIM;
        threadCount[1] = (int)check;
    }
    // printf("ThreadCount: %d, %d\n", threadCount[0], threadCount[1]);
}

__global__ void buildTree(Node* device_arr, Tree_Stats* device_tstats, int num_sub_tree_nodes, int board_size, curandState_t* rnd_states, size_t height, size_t width, size_t nodeArrSize)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    if(threadIdx.x == 31)
    {
        printf("THREADIDX.X = %d\n", threadIdx.x);
        printf("THREADIDX.Y = %d\n", threadIdx.y);
        printf("IDX = %d\n", idx);
    }
        
    int curr_node = 0;
    
    // printf("Test %d\n", idx);
    while(curr_node < num_sub_tree_nodes) // curr_node < (height-4) && idx < num_sub_tree_nodes
    {
        int arr_idx = idx*width + curr_node;//curr_node+width*idx;
        
        // Node* currentNode = &device_arr[arr_idx];
        
        if(device_arr[arr_idx].isReal)
        {
            for (int i = 0; i < 4; i++)
            {
                // printf("bs: %d\n", this.boardSize);
                GameState newState = GameState(board_size);
                print_board(device_arr[arr_idx].current_state);
                newState.copy(device_arr[arr_idx].current_state); //why?

                cuda_process_action(&newState, i, board_size);
                
                if(!determine_2048(device_arr[arr_idx].current_state) && !compare_game_states(device_arr[arr_idx].current_state, &newState))
                {
                    bool fullBoard = !cuda_add_new_number(&newState, rnd_states, &num_sub_tree_nodes);
                    if(!fullBoard)
                    {
                        int currentDepth = device_arr[arr_idx].depth + 1;
                        // if(device_tstats.max_depth < currentDepth)
                        //     device_tstats.max_depth = currentDepth;

                        // device_arr[arr_idx].children[i] = new Node(device_arr[arr_idx], newState, currentDepth);
                        Node newNode(&device_arr[arr_idx], &newState, currentDepth);
                        int new_arr_idx = (4*arr_idx+(i+1));
                        device_arr[new_arr_idx] = newNode;
                        
                        device_arr[arr_idx].children[i] = &device_arr[new_arr_idx];
                        // tree.num_nodes++;
                        print_board(&newState);                    
                        // device_tstats.num_nodes++;
                        device_arr[arr_idx].hasChildren = true;
                    }
                    else
                    {
                        device_arr[arr_idx].children[i] = nullptr;
                        Node newNode = Node();
                        int new_arr_idx = 4*arr_idx+(i+1);
                        device_arr[new_arr_idx] = newNode;
                    }
                }
                else
                {
                    device_arr[arr_idx].children[i] = nullptr;
                    Node newNode = Node();
                    int new_arr_idx = 4*arr_idx+(i+1);
                    device_arr[new_arr_idx] = newNode;
                }
                
    //             if(determine_2048(currentNode.current_state)) //win and shortest path
    //             {
    //                 if(device_tstats->optimal2048)
    //                 {
    //                     if(currentNode->depth < device_tstats->optimal2048->depth) 
    //                         device_tstats->optimal2048 = currentNode;
    //                 }
    //                 else
    //                     device_tstats->optimal2048 = currentNode;

    //                 device_tstats->num_solutions++;
    //             }

    //             if(determine_2048(currentNode->current_state) || compare_game_states(currentNode->current_state, newState)) 
    //             {
    //                 device_tstats->num_leaves++;
    //             }

    //             if(!determine_2048(currentNode->current_state) && !compare_game_states(currentNode->current_state, newState)) 
    //             {
    //                 device_tstats->num_cutoff_states++;
    //             }  
            } 
        }
        curr_node++;
        // __syncthreads();
    }
    __syncthreads();
}

__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    curand_init(seed, idx, 0, &states[idx]);
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