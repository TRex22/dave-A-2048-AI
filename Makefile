gamemanager:
	g++ -fopenmp -std=c++11 src/gamemanager/cpp/main.cpp -o bin/2048.out

clean:
	rm bin/2048.out