/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985
	
	Cuda Ai
*/

#define CUDA True //this is to use the same library functions
// #define BOARD_SIZE 4 

#include "../helper/helper.h"
#include "cuda_ai.h" //has all globals
#include "cuda_2048.cpp"
#include "serial_tree_builder.cpp"

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
    size_t width = (num_sub_tree_nodes)+1;    
    size_t nodeArrSize = height*width *sizeof(Node);
    size_t boardsArrSize = height*board_size*board_size;
    size_t mboardsArrSize = height*board_size*board_size*sizeof(int);
    size_t resultSize = width*board_size*board_size*sizeof(int);
    // printf("%d\n", height*width);                                                                                                             
    if(print_output)
        printf("Allocate host arr...\n");
    Node* host_arr = new Node[height*width];
    int* host_boards = (int*)malloc(mboardsArrSize);//new int[boardsArrSize]; //TODO deallocate this memory at end 
    for (unsigned int i = 0; i < boardsArrSize; ++i)
    {
        host_boards[i] = 0;
    }
    
    if(print_output)
        printf("Building initial tree...\n");

    std::stack<Node*> init_states;
    init_states = get_init_states(num_host_leaves); //gets all the cut off nodes for gpu
    
    for(unsigned int i = 0; i < height;i++)
    {
        host_arr[i*width] = *init_states.top();
        // printf("Tests: %i\n", i);

        for (int j = 0; j < board_size; j++)
        {
            for (int k = 0; k < board_size; k++)
            {
                // printf("Tests: %d\n", host_arr[i*width].current_state->currentBoard[j][k]);
                host_boards[i*board_size*board_size+j+(j*k)] = host_arr[i*width].current_state->currentBoard[j][k];
                // printf("%d\n", host_arr[i*width].current_state->currentBoard[j][k]);
            }
        }
        
        init_states.pop();
    }
    
//     int board_idx = 0*board_size*board_size;

//     for (int i = 0; i < board_size; ++i){
//         for (int j = 0; j < board_size; ++j)
//             printf("%d\n", host_boards[board_idx+i+(i*j)]);
//     }
    
    //update tree stats
    update_tree_stats(tstats, tree->root, tree->optimal2048, 0, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, tree->num_cutoff_states);
    
    if(print_output)
        printf("Move host array to device...\n");
    
    // device variables
    Node* device_arr;
    int* device_boards;
    int* result;
    Tree_Stats* device_tstats;
    int* device_num_sub_tree_nodes;
    
    int threadCounts[2] = {0, 0};
    calc_thread_count(threadCounts, height);
    dim3 dimBlock( threadCounts[0], threadCounts[1], 1 );
	dim3 dimGrid( 1, 1 );
     
    checkCudaErrors(cudaMalloc((void**)&device_arr, nodeArrSize));
    checkCudaErrors(cudaMalloc((void**)&device_boards, mboardsArrSize));
    checkCudaErrors(cudaMalloc((void**)&result, resultSize));
    checkCudaErrors(cudaMalloc((void**)&device_tstats, sizeof(Tree_Stats)));
    checkCudaErrors(cudaMalloc((void**)&device_num_sub_tree_nodes, sizeof(int)));
    
//     copy in values
    checkCudaErrors(cudaMemcpy(device_boards, host_boards, mboardsArrSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_num_sub_tree_nodes, &max_num_nodes, sizeof(int), cudaMemcpyHostToDevice));
    
    if(print_output)
        printf("Start buildTree kernel...\n");
    
    //curand stuff
    unsigned long int seed;
    if(use_rnd)
        seed = time(NULL);
    else
        seed = 10000;
    
    curandState_t* rnd_states;
    cudaMalloc((void**) &rnd_states, threadCounts[0]*threadCounts[1]*sizeof(curandState_t));
    
    //height/threadCounts[0], height
    init_rnd<<<dimGrid, dimBlock>>>(seed, rnd_states, device_num_sub_tree_nodes);
    build_trees<<<dimGrid, dimBlock>>>(device_arr, device_boards, result, num_sub_tree_nodes, board_size, rnd_states, height, width);
    
    if(print_output)
        printf("Copy results back to host...\n\n");

    // checkCudaErrors(cudaMemcpy(tstats, device_tstats, sizeof(Tree_Stats), cudaMemcpyDeviceToHost));
    
    float end_epoch = sdkGetTimerValue(&timer);
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        // print_solution(tree);
    }
    
    if(print_output)
    {
        printf("board_size: %i, num_nodes: %d, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, tstats->num_nodes, tstats->max_depth, tstats->num_solutions, tstats->num_leaves, ((double)tstats->num_solutions/(double)tstats->num_leaves));
        
        // if(tstats->optimal2048)
        //     printf("min_depth: %d time_taken: %f\n", tstats->optimal2048->depth, time_taken);
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
    // checkCudaErrors(cudaFree(device_boards)); //todo
    // checkCudaErrors(cudaFree(device_tstats));
    // checkCudaErrors(cudaFree(device_num_sub_tree_nodes));
}

__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void build_trees(Node* device_arr, int* device_boards, int* result, int num_sub_tree_nodes, int board_size, curandState_t* rnd_states, size_t height, size_t width)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;       
    int curr_node_idx = 0;
    int arr_idx = idx*width + curr_node_idx;
    int num_nodes = 1;
    
//     Init root from boards
    int board_idx = idx*board_size*board_size;
    
    //todo fix 4
    int** currentBoard = new int*[board_size];
    for (int i = 0; i < board_size; ++i){
        currentBoard[i] = new int[board_size];
        for (int j = 0; j < board_size; ++j)
        {
           currentBoard[i][j] = device_boards[board_idx+i+(i*j)];
        }
    }
    
    int new_arr_idx = 0;
       
    while(num_nodes < num_sub_tree_nodes)
    {
        arr_idx = idx*width + curr_node_idx;
        
        GameState currState(board_size, currentBoard);
        Node curr_node(nullptr, &currState, 0);
        device_arr[arr_idx] = curr_node;
        
        if(idx == 0)
        {
            // print_board(device_arr[arr_idx].current_state);
            print_board(&currState);
        }
        
        if(device_arr[arr_idx].isReal)
        {
            for (int i = 0; i < 4; i++)
            {
                num_nodes++;
                // printf("bs: %d\n", this.boardSize);
                GameState newState(board_size);               
                newState.copy(device_arr[arr_idx].current_state);

                cuda_process_action(&newState, i, board_size);
                
                int new_arr_idx = (4*arr_idx+(i+1));
                
                if(!determine_2048(device_arr[arr_idx].current_state) && !compare_game_states(device_arr[arr_idx].current_state, &newState))
                {
                    // printf("nes\n");
                    bool fullBoard = !cuda_add_new_number(&newState, rnd_states, &num_sub_tree_nodes);
                    int currentDepth = device_arr[arr_idx].depth + 1;
                    // if(device_tstats.max_depth < currentDepth)
                    //     device_tstats.max_depth = currentDepth;

                    Node newNode(&device_arr[arr_idx], &newState, currentDepth);
                    device_arr[new_arr_idx] = newNode;

                    device_arr[arr_idx].children[i] = &device_arr[new_arr_idx];
                    // tree.num_nodes++;

                    if(idx == 0)
                    {
                        printf("curr_node_idx: %d, arr_idx: %d, new_arr_idx: %d, num_nodes: %d\n", curr_node_idx, arr_idx, new_arr_idx, num_nodes);
                        // print_board(device_arr[arr_idx].current_state);
                        print_board(&newState);
                    }

                    // device_tstats.num_nodes++;
                    device_arr[arr_idx].hasChildren = true;
                }
                else
                {
                    device_arr[arr_idx].children[i] = nullptr;
                    Node newNode = Node();
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
        
        curr_node_idx++;
        currentBoard = device_arr[arr_idx].current_state->currentBoard;
    }
    
    printf("Num Nodes: %d thread: %uli\n", curr_node_idx, idx);

    // if(idx == 2)
    //     cuda_print_board(device_arr[arr_idx].current_state->currentBoard, board_size);

    // __syncthreads();
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
