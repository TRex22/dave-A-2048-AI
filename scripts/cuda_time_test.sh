#!/bin/bash
# board size
# height
# num nodes

#num_nodes,time_taken,board_size,height,width

make cuda

for b in 2 6 8
do
   echo "board_size: $i "
   
   for h in 1 2 100 1000 10000 100000 1000000
   do
       echo "height(num_trees): $h "
       
       for n in 10000 50000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 2000000 3000000 4000000 5000000
       do
           for repeat in {1..6}
           do
               w=$((n / h))
               echo "width(num_depth): $w "
               
               ./bin/cuda_ai.out --use_rnd --save_time --num_trees=$h --max_depth=$w 
           done
       done       
   done   
done