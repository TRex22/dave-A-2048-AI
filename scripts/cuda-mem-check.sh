#!/bin/bash
/usr/local/cuda/bin/nvcc src/cuda_ai/main.cu -I "/usr/local/cuda/samples/common/inc" -o bin/cuda1_ai.out -std=c++11 -lineinfo
/usr/local/cuda/bin/cuda-memcheck ./bin/cuda1_ai.out --print_output --print_path