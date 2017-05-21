There is a Makefile in the root folder which takes in either "serial", "cuda", "mpi" or "clean".
Clean removes the binary files in /bin.

There are bash scripts in the /scripts folder. These compile and run the binary files in /bin with some cmd line arguments. These are:
run_serial.sh
run_mpi.sh
cuda-mem-check.sh

CMD Line arguments:
--board_size = <board size> .This defines the board size to use.
--use_rnd . This seeds rand(). If not included then rand() is seeded using a constatnt value.
--max_depth = <depth> .This defines the depth the tree must be created until. Optional
--max_num_nodes = <nodes> .This defines the maximum number of nodes the tree must be generated until. 
--save_to_file .Saves the output to a file. If --save_to_csv is used with it then output is saved to a .csv file.
--filepath = <path to file> .this defines the path where the file should be saved to.
--print_output .This prints all output to terminal.
--print_path .This prints out the solution path to terminal.
DEBUG .Prints all debugging information to terminal.
--usage .Displays usage of cmd line inputs.