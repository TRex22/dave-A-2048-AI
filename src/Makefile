game:
	g++ -fopenmp -std=c++11 gamemanager/cpp/main.cpp -o ../bin/2048.out

cuda:
	/usr/local/cuda/bin/nvcc cuda1_ai.cu -I "/usr/local/cuda/samples/common/inc" -o ../bin/cuda1_ai.out -std=c++11

clean:
	rm ../bin/2048.out
	rm ../bin/cuda1_ai.out