#!/bin/bash 
set -e
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute prepare_from_excape.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute extract_actives.ipynb
bash ./train_all_models.sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute extract_best_model_and_save_final_model.ipynb
