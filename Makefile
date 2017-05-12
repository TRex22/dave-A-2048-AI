serial:
	g++ -fopenmp -std=c++11 src/serial_ai/main.cpp -o bin/serial_ai.out

game:
	g++ -fopenmp -std=c++11 src/gamemanager/cpp/main.cpp -o bin/2048.out

cuda:
	/usr/local/cuda/bin/nvcc src/cuda_ai/cuda1_ai.cu -I "/usr/local/cuda/samples/common/inc" -o bin/cuda1_ai.out -std=c++11

mpi:
	mpic++ src/mpi_si.main.cpp -o bin/mpi_ai.out

clean:
	rm bin/2048.out
	rm bin/cuda1_ai.out
	rm bin/serial_ai.out
	rm bin/mpi_ai.out

	rm ../results/serial_ai.txt
    rm ../results/serial_ai.csv
    