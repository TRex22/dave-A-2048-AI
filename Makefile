serial:
	g++ -fopenmp -std=c++11 src/serial_ai/main.cpp -o ../bin/serial_ai.out
gamemanager:
	g++ -fopenmp -std=c++11 src/gamemanager/cpp/main.cpp -o bin/2048.out

clean:
	rm bin/2048.out