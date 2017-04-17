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



class GameState {
private:
	

public:
	int boardSize;
	int score = 0;
	int** currentBoard = 0;
	int stepCount = 0;

	GameState(int);
	bool equals(GameState *state);
	void copy(GameState *state);


};

GameState::GameState(int board_size){
	this->boardSize = board_size;

	currentBoard = new int*[this->boardSize];
	for (int i = 0; i < this->boardSize; ++i)
	{
		currentBoard[i] = new int[this->boardSize];
	}
}

bool GameState::equals(GameState *state){
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

void GameState::copy(GameState *state){
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