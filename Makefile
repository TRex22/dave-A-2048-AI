serial:
	g++ -fopenmp -std=c++11 src/serial_ai/main.cpp -o bin/serial_ai.out -Wall -Wextra

game:
	g++ -fopenmp -std=c++11 src/gamemanager/cpp/main.cpp -o bin/2048.out -Wall -Wextra

cuda:
	/usr/local/cuda/bin/nvcc src/cuda_ai/main.cu -I "/usr/local/cuda/samples/common/inc" -o bin/cuda_ai.out -std=c++11 --compiler-options -Wall

mpi:
	mpic++ -fopenmp src/mpi_ai/main.cpp -o bin/mpi_ai.out -std=c++11 -Wall -Wextra

clean:
	rm bin/2048.out
	rm bin/cuda1_ai.out
	rm bin/serial_ai.out
	rm bin/mpi_ai.out

	rm ../results/serial_ai.txt
	rm ../results/serial_ai.csv

	rm ../results/mpi_ai.txt
	rm ../results/mpi_ai.csv

	rm ../results/cuda_ai.txt
	rm ../results/cuda_ai.csv
