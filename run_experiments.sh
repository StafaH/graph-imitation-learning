#!/bin/bash


NUM_RUNS=5
BASE_CMD="python src/graphs/reach_target_easy.py --tag fix_64x3_lr --hidden_dims 64 64 64 --data_dir data/reach_target_simple/ --model_name mlp --num_epochs 5000 --lr 0.0001 --eval_when_train"

for i in {1...$NUM_RUNS}
do 
    CMD="$BASE_CMD --seed $i"
    eval $CMD
done 
