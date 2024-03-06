#!/bin/bash

numgpus=$(python3 gpucount.py)

for i in $(seq 1 $numgpus); do
    CUDA_VISIBLE_DEVICES=$(( $i-1 )) python3 main.py $i $numgpus &
done
wait