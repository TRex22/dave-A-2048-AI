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

void write_results_to_file (std::string filename, std::string results)
{
	ofstream file;
	file.open(filename.c_str(), ios::app);
		file << results;
	file.close();
}

void write_results_to_file (std::string filename, std::string header, std::string results)
{
	ofstream file;
	file.open(filename.c_str(), ios::app);
		file << header << results << endl;
	file.close();
}

void print_system_specs()
{
	/*printf("Number of Processors: %d, Total Threads: %d\n---\n\n", omp_get_num_procs(), omp_get_max_threads());*/
}

void print_cmd_heading(string app_name)
{
	// usually dont use printf with c++, use cout but due to cout not being thread safe and this
	// assignment is about parallelism so I will not mix printf and cout as mixing techniques is 
	// bad software etiquette.
	printf("%s\nJason Chalom 711985\n2017\n\n", app_name.c_str());
	print_system_specs();
}

