#!/bin/bash -i

if [ "$1" == "0" ]; then
    TASKS="yelp_review_full_csv ag_news_csv dbpedia_csv amazon_review_full_csv yahoo_answers_csv"
elif [ "$1" == "1" ]; then
    TASKS="dbpedia_csv yahoo_answers_csv ag_news_csv amazon_review_full_csv yelp_review_full_csv"
elif [ "$1" == "2" ]; then
    TASKS="yelp_review_full_csv yahoo_answers_csv amazon_review_full_csv dbpedia_csv ag_news_csv"
elif [ "$1" == "3" ]; then
    TASKS="ag_news_csv yelp_review_full_csv amazon_review_full_csv yahoo_answers_csv dbpedia_csv"
fi

# ./run_v3.sh 0 MetaMBPA_order1_v3
# (1) Adaptation 32 x 20 times
# (3) Write 1%
# (4) Change Loss Calc
# For run v3.2
python3 train_v3.py --tasks $TASKS --output_dir "/data/model_runs/em_in_lll/$2" --write_prob 0.01 --inner_lr 1e-5 --learning_rate 3e-5 
# For run v3.3
# python3 train_v3.py --tasks $TASKS --output_dir "/data/model_runs/em_in_lll/$2" --write_prob 0.01 --inner_lr 5e-5 --learning_rate 3e-5 