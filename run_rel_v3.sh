#!/bin/bash -i

# ./run_rel_v3.sh MetaMBPA_rel
python3 train_rel_v3.py --output_dir "/data/model_runs/em_in_lll/$1" --write_prob 0.01 --inner_lr 1e-5 --learning_rate 3e-5 --batch_size 4
#  Shuffle Index: [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
# [TIME] Training Time within 37.63943734884262 hours