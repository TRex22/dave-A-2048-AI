/*
	2048 Game for HPC Group Assignment
	Liron Mizrahi 708810
	Jason Chalom 711985

	This is the serial AI approach

	boardSize = since the board is square this is just the width of the board
	score = current player score, starts at 0
	currentBoard = current squares starts at 0 and has all the numbers on the board
	

	AI General Algorithm:
	1. Generate a random init board
	2. loop (in ai using these functions) until 2048
	3. in loop:
		a. guess a move
		b. check new gamestate
		c. retry if fail
		d. go back to a if pass
	4.yay win!
*/