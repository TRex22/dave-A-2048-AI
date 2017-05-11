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

void save_solution_to_file(Tree* tree, Node* a2048, float time_taken, std::string filename);
string flatten_path(Tree* tree, Node* final_node);
string str_node(GameState* currentGame);

void print_left_most_path(Tree* tree);
void print_right_most_path(Tree* tree);
bool print_solution(Tree* tree);

bool checkAtRoot(Node* node);
bool canContinue(Node* node);
bool shouldLimit(Tree* tree, int depth_limit, int node_limit);

void halt_execution(string message);
void write_results_to_file (std::string filename, std::string results);
void write_results_to_file (std::string filename, std::string header, std::string results);
void print_cmd_heading(string app_name);


#include "../helper/helper.cpp"