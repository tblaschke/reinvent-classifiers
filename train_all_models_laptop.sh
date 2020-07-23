#!/bin/bash

for kernel in minmax tanimoto; do
   for target in DRD2 HTR1A; do
       
      echo "Precompute Gram"
      ./precompute_gram_args.py ${target} ${kernel}
      echo "" > jobs
      for bal in balanced unbalanced; do
        for c in 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000; do 
          echo "./train_models_args.py ${target} ${c} ${kernel} ${bal} r" >> jobs 
        done
      done
      echo "Fit SVMs"
      parallel < jobs
      rm jobs
      if [[ "${kernel}" != "minmax" ]]; then
          rm -f ${target}_${kernel}_training_X.npy
          rm -f ${target}_${kernel}_training_Y.npy
          rm -f ${target}_${kernel}_test_X.npy
          rm -f ${target}_${kernel}_test_Y.npy
          rm -f ${target}_${kernel}_validation_X.npy
          rm -f ${target}_${kernel}_validation_Y.npy
      fi
   done 
done
