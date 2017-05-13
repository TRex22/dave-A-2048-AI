#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stack>

#include "helper/helper.h"

using namespace std;

int main(int argc, char const *argv[])
{
	GameState* state = new GameState(4);
	printf("state:\n");
    	print_board(state);

    printf("state after adding number:\n");
		add_new_number(state);
		print_board(state);
	return 0;
}