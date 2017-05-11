#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>
#include <random>

// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

/*headers*/


using namespace std;

void halt_execution(string message)
{
	cudaDeviceReset();
	cout << message << endl;
    exit(EXIT_FAILURE);
}

void print_system_specs()
{
	/*printf("Number of Processors: %d, Total Threads: %d\n---\n\n", omp_get_num_procs(), omp_get_max_threads());*/
}

