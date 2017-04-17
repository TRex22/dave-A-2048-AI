/*
	Helper functions
	Jason Chalom 711985
	2017
*/

#include "experiment_helper.cpp"

void write_results_to_file (std::string filename, std::string results);
void write_results_to_file (std::string filename, std::string header, std::string results);
void print_system_specs();
void print_cmd_heading(string app_name);

void halt_execution(string message);