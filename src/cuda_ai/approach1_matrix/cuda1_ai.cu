/*
	HPC Assignment 2 Naive Parallel Implementation
	Jason Chalom 711985
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../headers/2048.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

#define heading "CUDA Dave Ai for playing 2048 using matrix approach"
#define results_header "children,nodes,time,win rate,flops"
#define results_location "../results/results_cuda1_ai.csv"
#define DIM 1024
#define warp 32

/*Global Variables*/
int MAX_CHILDREN = 4;
int MAX_NODES = 2048; //experiment with values
int MAX_TIME = 50; //in ms
int MAX_GAME_SIZE = 4;

// Auto-Verification Code
bool testResult = true;

using namespace std;

/*Function Headers*/
void run_test(char *argv[]);
char** initTreeArray(int num_nodes, int num_children);

int main(int argc, char *argv[])
{
	//some basic setup
	print_cmd_heading(heading);

    if (argc > 1)
    {
    	if(checkCmdLineFlag(argc, (const char **) argv, "run_test"))
		{
			run_test(argv);

			cudaDeviceReset();
		    printf("%s completed, returned %s\n",
		           heading,
		           testResult ? "OK" : "ERROR!");
		    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
		}
		else
		{
			//todo JMC: add checks for non ints
	        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
	        {
	        	char node;
	            getCmdLineArgumentString(argc,
	                                     (const char **) argv,
	                                     "n",
	                                     (char **) &node);
	            MAX_NODES = node - '0';
	        }
	        
	        if (checkCmdLineFlag(argc, (const char **) argv, "c"))
	        {
	        	char child;
	            getCmdLineArgumentString(argc,
	                                     (const char **) argv,
	                                     "c",
	                                     (char **) &child);
	            MAX_CHILDREN = child - '0';
	        }
	        
	        if (checkCmdLineFlag(argc, (const char **) argv, "t"))
	        {
	        	char time;
	            getCmdLineArgumentString(argc,
	                                     (const char **) argv,
	                                     "t",
	                                     (char **) &time);
	            MAX_TIME = time - '0';
	        }
		}
    }
	else
	{
		printf("options: -n (number of nodes)\n");
		printf("options: -c (number of children)\n");
		printf("options: -t (max time)\n");
		printf("options: -run_test (run a batch test on generated data)\n");

		printf("usage: %s -n 2048 -c 4 -t 48\n", argv[0]);
		printf("usage: %s -run_test\n", argv[0]);
		halt_execution("");
	}

	else
	{
		cout << "Starting to play 2048....." << endl;

		char ** hostTreeArray = initTreeArray(MAX_NODES, MAX_CHILDREN);

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

		cudaDeviceReset();
	    printf("%s completed, returned %s\n",
	           heading,
	           testResult ? "OK" : "ERROR!");
	    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
	}
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

//helper functions
char** initTreeArray(int num_nodes, int num_children)
{
	char *treeArr = ( *) malloc(size)
}

void run_test(char *argv[])
{
	cout << "Starting to Run Batch Experiments....." << endl;
	int max_n = 3000;

	write_results_to_file(results_location, results_header, "");

	/*warmup round*/


	/*cuda vars*/


	//todo JMC: get this done so we can run some tests
}