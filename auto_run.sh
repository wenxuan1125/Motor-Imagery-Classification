#!/bin/sh

batch_size='4 8 16 32'
lr='5e-2 1e-2 5e-3 1e-3 5e-4 1e-4'
gamma='0.9 0.85 0.8 0.75 0.7 0.65'

for i in $batch_size; do
    for j in $lr; do
        for k in $gamma; do
            # echo "$i $j $k"
            python read_experiment.py --batch-size=$i --lr=$j --gamma=$k
        done
    done
done