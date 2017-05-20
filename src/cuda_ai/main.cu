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
    cudaDeviceReset();
    
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
      
    size_t height = num_host_leaves;//(max_num_nodes/width); 
    size_t width = max_depth;//(num_sub_tree_nodes);
    
    size_t nodeArrSize = height*width *sizeof(Node);
    size_t boardsArrSize = height*board_size*board_size;
    size_t mboardsArrSize = height*board_size*board_size*sizeof(int);
    size_t resultSize = width*board_size*board_size*sizeof(int);
    // printf("%d\n", height);
    
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
    init_states = get_init_states(num_host_leaves); // num_host_leaves height//gets all the cut off nodes for gpu
    
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
    
    //update tree stats
    update_tree_stats(tstats, tree->root, tree->optimal2048, 0, tree->num_nodes, tree->max_depth, tree->num_solutions, tree->num_leaves, tree->num_cutoff_states);
    
    if(print_output)
        printf("Move host array to device...\n");
    
    // device variables
    Node* device_arr;
    int* device_boards;
    int*** result;
    Tree_Stats* device_tstats;
    int* device_num_sub_tree_nodes;
    
    int threadCounts[2] = {0, 0};
    calc_thread_count(threadCounts, height);
    dim3 dimBlock( threadCounts[0], threadCounts[1], 1 );
	dim3 dimGrid( 1, 1 );
     
    checkCudaErrors(cudaMalloc((void**)&device_arr, nodeArrSize));
    checkCudaErrors(cudaMalloc((void**)&device_boards, mboardsArrSize));
    checkCudaErrors(cudaMalloc((void****)&result, resultSize));
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
    build_trees<<<dimGrid, dimBlock>>>(device_arr, device_boards, device_tstats, result, num_sub_tree_nodes, board_size, rnd_states, height, width, print_output, DEBUG);
    
    if(print_output)
        printf("Copy results back to host...\n\n");
    
    // int*** h_result = (int***)malloc(resultSize);
    // checkCudaErrors(cudaMemcpy(h_result, result, resultSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tstats, device_tstats, sizeof(Tree_Stats), cudaMemcpyDeviceToHost));
    
    float end_epoch = sdkGetTimerValue(&timer);
    time_taken = end_epoch-start_epoch;
    
    if(print_path)
    {
        // print_solution(tree);
    }
    
    if(print_output)
    {
        printf("board_size: %i, num_nodes: %lu, max_depth: %d, sols: %d, leaves: %d, stats: %f\n", board_size, height*width, tstats->max_depth, tstats->num_solutions, tstats->num_leaves, ((double)tstats->num_solutions/(double)tstats->num_leaves));
        printf("time_taken: %f\n", time_taken);
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
    checkCudaErrors(cudaFree(device_arr));
    checkCudaErrors(cudaFree(device_boards));
    checkCudaErrors(cudaFree(result)); //todo
    checkCudaErrors(cudaFree(device_tstats));
    checkCudaErrors(cudaFree(device_num_sub_tree_nodes));
}

__global__ void init_rnd(unsigned int seed, curandState_t* states, int* device_num_sub_tree_nodes) {
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void build_trees(Node* device_arr, int* device_boards, Tree_Stats* device_tstats, int*** result, int num_sub_tree_nodes, int board_size, curandState_t* rnd_states, size_t height, size_t width, bool print_output, bool DEBUG)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;       
    int curr_node_idx = 0;
    int arr_idx = idx*width + curr_node_idx;
    int num_nodes = 1;
    
    // printf("height: %d\n", height);
    
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
    
    if(idx == 0)
    {
        device_tstats->num_leaves = 0;
        device_tstats->num_cutoff_states = 0;
        device_tstats->max_depth = 0;
    }
    
    Node* localTree = new Node[width];
       
    while(num_nodes < width)
    {
        arr_idx = idx*width + curr_node_idx;
        
        GameState currState(board_size, currentBoard);
        Node curr_node(nullptr, &currState, 0);
        // device_arr[arr_idx] = curr_node;
        localTree[num_nodes] = curr_node;
        
        if(idx == 0 && DEBUG)
        {
            // print_board(device_arr[arr_idx].current_state);
            print_board(&currState);
        }
        
        if(localTree[num_nodes].isReal)//device_arr[arr_idx].isReal)
        {
            for (int i = 0; i < 4; i++)
            {
                num_nodes++;
                GameState newState(board_size);               
                // newState.copy(device_arr[arr_idx].current_state);
                newState.copy(localTree[num_nodes].current_state);

                cuda_process_action(&newState, i, board_size);
                
                // new_arr_idx = (4*arr_idx+(i+1));
                new_arr_idx = (4*num_nodes+(i+1));
                
                //if(!determine_2048(device_arr[arr_idx].current_state) && !compare_game_states(device_arr[arr_idx].current_state, &newState))
                if(!determine_2048(localTree[num_nodes].current_state) && !compare_game_states(localTree[num_nodes].current_state, &newState))
                {
                    bool fullBoard = !cuda_add_new_number(&newState, rnd_states, &num_sub_tree_nodes);
                    // int currentDepth = device_arr[arr_idx].depth + 1;
                    int currentDepth = localTree[num_nodes].depth + 1;
                    
                    if(idx == 0)
                    {
                        if(device_tstats->max_depth < currentDepth)
                            device_tstats->max_depth = currentDepth;
                    }

                    // Node newNode(&device_arr[arr_idx], &newState, currentDepth);
                    // device_arr[new_arr_idx] = newNode;
                    Node newNode(&localTree[num_nodes], &newState, currentDepth);
                    localTree[new_arr_idx] = newNode;
                    

                    // device_arr[arr_idx].children[i] = &device_arr[new_arr_idx];
                    localTree[num_nodes].children[i] = &localTree[new_arr_idx];

                    if(idx == 0 && DEBUG)
                    {
                        printf("curr_node_idx: %d, arr_idx: %d, new_arr_idx: %d, num_nodes: %d\n", curr_node_idx, num_nodes, new_arr_idx, num_nodes);
                        // print_board(device_arr[arr_idx].current_state);
                        print_board(&newState);
                    }

                    // device_arr[arr_idx].hasChildren = true;
                    localTree[num_nodes].hasChildren = true;
                }
                else
                {
                    // device_arr[arr_idx].children[i] = nullptr;
                    localTree[num_nodes].children[i] = nullptr;
                    Node newNode = Node();
                    // device_arr[new_arr_idx] = newNode;
                    localTree[new_arr_idx] = newNode;
                }
                

                // if(idx == 0)
                //     if(determine_2048(device_arr[arr_idx].current_state) || compare_game_states(device_arr[arr_idx].current_state, &newState)) 
                // {
                //     device_tstats->num_leaves++;
                // }
                // if(idx == 0)
                //     if(!determine_2048(device_arr[arr_idx].current_state) && !compare_game_states(device_arr[arr_idx].current_state, &newState)) 
                // {
                //     device_tstats->num_cutoff_states++;
                // }  
                
                if(idx == 0)
                    if(determine_2048(localTree[num_nodes].current_state) || compare_game_states(localTree[num_nodes].current_state, &newState)) 
                {
                    device_tstats->num_leaves++;
                }
                if(idx == 0)
                    if(!determine_2048(localTree[num_nodes].current_state) && !compare_game_states(localTree[num_nodes].current_state, &newState)) 
                {
                    device_tstats->num_cutoff_states++;
                }  
            } 
        }
        
        curr_node_idx++;
        // currentBoard = device_arr[arr_idx].current_state->currentBoard;
        currentBoard = localTree[num_nodes].current_state->currentBoard;
        device_tstats->num_nodes += num_nodes;
    }

    __syncthreads();
    
    
//     Find solution
//     if(idx == 0)
//     {
//         Node optimal2048(nullptr, nullptr, width);
//         optimal2048.isReal = false;
//         device_tstats->num_solutions = 0;
        
//         for(int i = (height*width)-1; i >=0; i--)
//         {
//             if(device_arr[i].isReal)
//                 if(determine_2048(device_arr[i].current_state))
//                 {
//                     device_tstats->num_solutions++;
//                     // printf("Solution!\n");
//                     if(device_arr[i].depth < optimal2048.depth)
//                         optimal2048 = device_arr[i];
//                 }
//         }
        
//         // write to results arr
//         // size_t resultSize = width*board_size*board_size*sizeof(int);
//         if(optimal2048.isReal)
//         {
//             result[optimal2048.depth] = optimal2048.current_state->currentBoard;
//             for (int i = optimal2048.depth-1; i >= 0; i++)
//             {
//                 Node curr_node = *optimal2048.parent;
//                 result[i] = curr_node.current_state->currentBoard;
//             }
//         }
        
//     }
    
    if(idx == 0)
    {
        device_tstats->num_nodes = height*width;
        // printf("Solution: %d\n", device_tstats->num_solutions);
    }
    
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
