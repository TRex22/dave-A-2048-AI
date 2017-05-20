#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>
#include <random>

#include "../tree/tree.h"

using namespace std;

void save_time_to_file(int h, int w, int b, float time_taken);
void save_solution_to_file(Tree* tree, float time_taken, std::string filename, bool csv);
string flatten_path(Tree* tree, Node* final_node, string delimit);
string str_node(GameState* currentGame, string delimit);

void print_left_most_path(Tree* tree);
void print_right_most_path(Tree* tree);
bool print_solution(Tree* tree);

bool checkAtRoot(Node* node);
bool canContinue(Node* node);
bool shouldLimit(Tree* tree, int depth_limit, int node_limit, float current_time, float time_limit);
bool shouldLimit(Tree* tree, int cutoff_node_limit);

void halt_execution(string message);
void write_results_to_file (std::string filename, std::string results);
void write_results_to_file (std::string filename, std::string header, std::string results);
void print_cmd_heading(string app_name);

void print_usage(int argc, char *argv[]);


#include "../helper/helper.cpp"