#!/bin/bash
set -e
echo "" > jobs
echo "Precompute Gram"
kernel=minmax
for target in DRD2 HTR1A; do
    echo "./precompute_gram_args.py ${target} ${kernel}" >> jobs
done
parallel < jobs
echo "" > jobs
echo "Fit SVMs"
echo "" > jobs
for target in DRD2 HTR1A; do
    for bal in balanced unbalanced; do
        for c in 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000; do 
          echo "./train_models_args.py ${target} ${c} ${kernel} ${bal} r" >> jobs 
        done
    done
done
parallel < jobs
rm jobs