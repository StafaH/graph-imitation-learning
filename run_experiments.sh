#!/bin/bash

NUM_RUNS=5
BASE_CMD=$1

for i in $(seq 1 $NUM_RUNS);
do 
    CMD="$BASE_CMD --seed $i"
    echo $CMD
    eval $CMD
done 
