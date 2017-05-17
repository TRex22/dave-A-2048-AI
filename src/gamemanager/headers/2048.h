/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the main game supervisor used in the ui and simulations of MCTS

	boardSize = since the board is square this is just the width of the board
	score = current player score, starts at 0
	currentBoard = current squares starts at 0 and has all the numbers on the board
	stepCount = 
*/

#include <random>
#include <math.h>
#include <string.h>
#include <string>

class GameState 
{
	private:
		

	public:
		int boardSize;
		int score = 0;
		int** currentBoard = 0;
		int stepCount = 0;
		bool isWon = false;
		bool invalidMove = false;

#ifdef CUDA    
    __host__ __device__
#endif
		GameState(int);
    
#ifdef CUDA 
    __host__ __device__
#endif
        GameState(int __boardSize, int** __currentBoard);
    
#ifdef CUDA 
    __host__ __device__
#endif
		bool equals(GameState *state);
    
#ifdef CUDA 
    __host__ __device__
#endif
		void copy(GameState *state);
    
#ifdef CUDA 
    __host__ __device__
#endif
		void init_to_zero(GameState* state);
};

#ifdef CUDA 
    __host__ __device__
#endif
GameState::GameState(int board_size)
{
	this->boardSize = board_size;

	currentBoard = new int*[this->boardSize];
	for (int i = 0; i < this->boardSize; ++i)
	{
		currentBoard[i] = new int[this->boardSize];
	}
	init_to_zero(this);
}

#ifdef CUDA 
    __host__ __device__
#endif
GameState::GameState(int __boardSize, int** __currentBoard)
{
	this->boardSize = __boardSize;
    this->currentBoard = __currentBoard;
}

#ifdef CUDA 
    __host__ __device__
#endif
bool GameState::equals(GameState *state)
{
	bool result = true;
	for (int i = 0; i < this->boardSize; ++i)
	{
		for (int j = 0; j < this->boardSize; ++j)
		{
			if (this->currentBoard[i][j] != state->currentBoard[i][j])
			{
				result = false;
				break;
			}
		}
	}
	return result;
}

#ifdef CUDA 
    __host__ __device__
#endif
void GameState::copy(GameState *state)
{
	this->boardSize = state->boardSize;
	this->score = state->score;
	this->stepCount = state->stepCount;

	for (int i = 0; i < this->boardSize; ++i)
	{
		for (int j = 0; j < this->boardSize; ++j)
		{
			this->currentBoard[i][j] = state->currentBoard[i][j];
		}
	}
}

#ifdef CUDA 
    __host__ __device__
#endif
void GameState::init_to_zero(GameState* state)
{
    for (int i = 0; i < state->boardSize; ++i)
    {
        for (int j = 0; j < state->boardSize; ++j)
        {
            state->currentBoard[i][j] = 0;
        }
    }
} 