__device__ bool cuda_add_new_number(GameState *currentGame, curandState_t* states, int* device_num_sub_tree_nodes)
{  
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    curandState localState = states[idx];
    
    int rand_row = curand(&localState) % currentGame->boardSize;
    int rand_col = curand(&localState) % currentGame->boardSize;
    
    // printf("%d\n",rand_row);
    
	if(checkBoardEmptySlot(currentGame))
	{
		while(currentGame->currentBoard[rand_row][rand_col] != 0)
		{
            rand_row = curand(&localState) % currentGame->boardSize;
            rand_col = curand(&localState) % currentGame->boardSize;
		}

		currentGame->currentBoard[rand_row][rand_col] = 2;
		return true;
	}
    
    states[idx] = localState;
    
	return false;
}

__device__ void cuda_process_action(GameState *currentGame, int action, int boardSize)
{
	if (action == 0)
	{
		cuda_process_left(currentGame, boardSize);
	}
	else if (action == 1)
	{
		cuda_process_right(currentGame, boardSize);
	}
	else if (action == 2)
	{
		cuda_process_up(currentGame, boardSize);
	}
	else if (action == 3)
	{
		cuda_process_down(currentGame, boardSize);
	}
}

__device__ void cuda_process_left(GameState *currentGame, int boardSize)
{
    bool* modified = (bool*)malloc(sizeof(bool)*(boardSize));
	for (int i = 0; i < boardSize; ++i)
	{
		for (int p = 0; p < boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int j = 1; j < boardSize; ++j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t > 0 && currentGame->currentBoard[i][t-1] == 0)
				{
					currentGame->currentBoard[i][t-1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[i][t-1] == currentGame->currentBoard[i][t] &&  modified[t-1] == false)
				{
					currentGame->currentBoard[i][t-1] += currentGame->currentBoard[i][t];
					modified[t-1] = true;
					currentGame->score += currentGame->currentBoard[i][t-1];
					currentGame->currentBoard[i][t] = 0;
				}
			}
			
		}
	}
    free(modified);
}

__device__ void cuda_process_right(GameState *currentGame, int boardSize)
{
    bool* modified = (bool*)malloc(sizeof(bool)*(boardSize));
	for (int i = 0; i < boardSize; ++i)
	{          
		for (int p = 0; p < boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int j = boardSize - 2; j > -1; --j)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = j;
				while(t < boardSize - 1 && currentGame->currentBoard[i][t+1] == 0)
				{
					currentGame->currentBoard[i][t+1] = currentGame->currentBoard[i][t];
					currentGame->currentBoard[i][t] = 0;
					t++;
				}

				if (t == boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[i][t+1] == currentGame->currentBoard[i][t] && modified[t+1] == false)
				{
					currentGame->currentBoard[i][t+1] += currentGame->currentBoard[i][t];
					modified[t+1] = true;
					currentGame->score += currentGame->currentBoard[i][t+1];
					currentGame->currentBoard[i][t] = 0;
				}
			}
		}
	}
    free(modified);
}

__device__ void cuda_process_up(GameState *currentGame, int boardSize)
{
    bool* modified = (bool*)malloc(sizeof(bool)*(boardSize));   
	for (int j = 0; j < boardSize; ++j)
	{          
		for (int p = 0; p < boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int i = 1; i < boardSize; ++i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t > 0 && currentGame->currentBoard[t-1][j] == 0)
				{
					currentGame->currentBoard[t-1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t--;
				}

				if (t == 0)
				{
					t++;
				}

				if (currentGame->currentBoard[t-1][j] == currentGame->currentBoard[t][j] &&  modified[t-1] == false)
				{
					currentGame->currentBoard[t-1][j] += currentGame->currentBoard[t][j];
					modified[t+1] = true;
					currentGame->score += currentGame->currentBoard[i][t-1];
					currentGame->currentBoard[t][j] = 0;
				}
			}
		}
	}
    free(modified);
}

__device__ void cuda_process_down(GameState *currentGame, int boardSize)
{
    bool* modified = (bool*)malloc(sizeof(bool)*(boardSize));   
	for (int j = 0; j < boardSize; ++j)
	{      
		for (int p = 0; p < boardSize; ++p)
		{
			modified[p] = false;
		}

		for (int i = boardSize - 2; i > -1; --i)
		{
			if (currentGame->currentBoard[i][j] != 0)
			{
				int t = i;
				while(t < boardSize - 1 && currentGame->currentBoard[t+1][j] == 0)
				{
					currentGame->currentBoard[t+1][j] = currentGame->currentBoard[t][j];
					currentGame->currentBoard[t][j] = 0;
					t++;
				}

				if (t == boardSize - 1)
				{
					t--;
				}

				if (currentGame->currentBoard[t+1][j] == currentGame->currentBoard[t][j] && modified[t+1] == false)
				{
					currentGame->currentBoard[t+1][j] += currentGame->currentBoard[t][j];
					modified[t+1] = true;
					currentGame->score += currentGame->currentBoard[t+1][j];
					currentGame->currentBoard[t][j] = 0;
				}
			}
		}
	}
    free(modified);
}

__device__ void cuda_print_board(int** currentBoard, int board_size)
{
	for (int q = 0; q < board_size; q++)
	{
		print_horizontal_boarder(board_size);

		for (int p = 0; p < board_size; p++)
		{
			printf("| \t%d\t |", currentBoard[q][p]);
		}

		print_horizontal_boarder(board_size);
		printf("\n");
	}
    printf("\n");
}