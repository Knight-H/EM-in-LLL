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

# ./run_v2.sh 0 MetaMBPA_order1
# python3 train_v2.py --tasks $TASKS --output_dir "/data/model_runs/em_in_lll/$2" --write_prob 0.1
# Actually wrong in that this is write prob 10%... 
# Learning rate for inner and outer is the same...
# And adaptation is SGD over 32 x 32 batch, instead of a batch of 32 adapted over 20 times!!!!! this is more like OML than meta-mbpa

# 2023-06-12 17:16:00,137 - 0:00:07 - 0.000s - INFO - __main__ - args: Namespace(adam_epsilon=1e-08, batch_size=12353, debug=False, device_id=0, learning_rate=2e-05, logging_steps=500, max_grad_norm=1.0, model_name='bert-base-uncased', model_type='bert', n_labels=33, n_neighbors=32, n_test=7600, n_train=115000, n_workers=4, output_dir='/data/model_runs/em_in_lll/MetaMBPA_order1', overwrite=False, replay_interval=100, reproduce=False, seed=42, tasks=['yelp_review_full_csv', 'ag_news_csv', 'dbpedia_csv', 'amazon_review_full_csv', 'yahoo_answers_csv'], valid_ratio=0, warmup_steps=0, weight_decay=0, write_prob=0.1)
# 2023-06-12 17:17:47,194 - 0:01:54 - 93.387s - INFO - __main__ - Start training yelp_review_full_csv...    
# 2023-06-12 18:43:08,966 - 1:27:16 - 5121.772s - INFO - __main__ - progress: 0.15 , step: 500 , lr: 1.91E-05 , avg batch size: 33.0 , avg loss: 3.606                                                                
# 2023-06-12 20:07:32,395 - 2:51:39 - 5063.430s - INFO - __main__ - progress: 0.29 , step: 1000 , lr: 1.82E-05 , avg batch size: 33.0 , avg loss: 3.609                                                               
# 2023-06-12 21:32:51,093 - 4:16:58 - 5118.697s - INFO - __main__ - progress: 0.44 , step: 1500 , lr: 1.74E-05 , avg batch size: 33.0 , avg loss: 3.609                                                               
# 2023-06-12 23:00:21,869 - 5:44:29 - 5250.776s - INFO - __main__ - progress: 0.58 , step: 2000 , lr: 1.65E-05 , avg batch size: 33.0 , avg loss: 3.610                                                               
# 2023-06-13 00:28:27,876 - 7:12:35 - 5286.007s - INFO - __main__ - progress: 0.73 , step: 2500 , lr: 1.56E$05 , avg batch size: 33.0 , avg loss: 3.610                                                               
# 2023-06-13 01:56:33,058 - 8:40:40 - 5285.182s - INFO - __main__ - progress: 0.88 , step: 3000 , lr: 1.47E$05 , avg batch size: 33.0 , avg loss: 3.610                                                               
# 2023-06-13 03:11:55,390 - 9:56:02 - 4522.332s - INFO - __main__ - Finsih training, avg loss: 3.611
# 2023-06-13 03:13:02,656 - 9:57:09 - 56.156s - INFO - __main__ - Start training ag_news_csv...             
# 2023-06-13 04:51:45,026 - 11:35:52 - 5922.369s - INFO - __main__ - progress: 0.20 , step: 500 , lr: 1.91E$05 , avg batch size: 46.0 , avg loss: 3.322                                                               
# 2023-06-13 06:20:34,935 - 13:04:42 - 5329.909s - INFO - __main__ - progress: 0.40 , step: 1000 , lr: 1.82$-05 , avg batch size: 46.0 , avg loss: 3.320                                                              
# 2023-06-13 07:47:53,341 - 14:32:00 - 5238.406s - INFO - __main__ - progress: 0.60 , step: 1500 , lr: 1.74$-05 , avg batch size: 46.0 , avg loss: 3.319                                                              
# 2023-06-13 09:13:05,726 - 15:57:13 - 5112.385s - INFO - __main__ - progress: 0.80 , step: 2000 , lr: 1.65$-05 , avg batch size: 45.0 , avg loss: 3.319                                                              
# 2023-06-13 10:41:02,512 - 17:25:09 - 5276.786s - INFO - __main__ - progress: 1.00 , step: 2500 , lr: 1.56$-05 , avg batch size: 45.0 , avg loss: 3.320                                                              
# 2023-06-13 10:42:25,970 - 17:26:33 - 83.458s - INFO - __main__ - Finsih training, avg loss: 3.320
# 2023-06-13 10:43:50,552 - 17:27:57 - 73.609s - INFO - __main__ - Start training dbpedia_csv...            
# 2023-06-13 12:21:38,301 - 19:05:45 - 5867.748s - INFO - __main__ - progress: 0.19 , step: 500 , lr: 1.91E-05 , avg batch size: 42.0 , avg loss: 3.514                                                               
# 2023-06-13 13:57:28,440 - 20:41:35 - 5750.139s - INFO - __main__ - progress: 0.37 , step: 1000 , lr: 1.82E-05 , avg batch size: 42.0 , avg loss: 3.510                                                              
# 2023-06-13 15:33:08,669 - 22:17:15 - 5740.230s - INFO - __main__ - progress: 0.56 , step: 1500 , lr: 1.74E-05 , avg batch size: 42.0 , avg loss: 3.511                                                              
# 2023-06-13 17:10:00,110 - 23:54:07 - 5811.441s - INFO - __main__ - progress: 0.74 , step: 2000 , lr: 1.65E-05 , avg batch size: 42.0 , avg loss: 3.511                                                              
# 2023-06-13 18:47:54,428 - 1 day, 1:32:01 - 5874.318s - INFO - __main__ - progress: 0.93 , step: 2500 , lr: 1.56E-05 , avg batch size: 42.0 , avg loss: 3.512                                                        
# 2023-06-13 19:26:25,068 - 1 day, 2:10:32 - 2310.639s - INFO - __main__ - Finsih training, avg loss: 3.512 
# 2023-06-13 19:28:32,926 - 1 day, 2:12:40 - 116.200s - INFO - __main__ - Start training amazon_review_full_csv...
# 2023-06-13 21:20:16,330 - 1 day, 4:04:23 - 6703.404s - INFO - __main__ - progress: 0.17 , step: 500 , lr: 1.91E-05 , avg batch size: 38.0 , avg loss: 3.587                                                         
# 2023-06-13 23:11:37,248 - 1 day, 5:55:44 - 6680.918s - INFO - __main__ - progress: 0.33 , step: 1000 , lr: 1.82E-05 , avg batch size: 38.0 , avg loss: 3.587                                                        
# 2023-06-14 01:03:41,525 - 1 day, 7:47:48 - 6724.277s - INFO - __main__ - progress: 0.50 , step: 1500 , lr: 1.74E-05 , avg batch size: 38.0 , avg loss: 3.588                                                        
# 2023-06-14 02:57:35,122 - 1 day, 9:41:42 - 6833.597s - INFO - __main__ - progress: 0.67 , step: 2000 , lr: 1.65E-05 , avg batch size: 38.0 , avg loss: 3.589                                                        
# 2023-06-14 04:52:36,780 - 1 day, 11:36:44 - 6901.658s - INFO - __main__ - progress: 0.84 , step: 2500 , lr: 1.56E-05 , avg batch size: 38.0 , avg loss: 3.592                                                       
# 2023-06-14 06:45:53,448 - 1 day, 13:30:00 - 6796.668s - INFO - __main__ - Finsih training, avg loss: 3.592
# 2023-06-14 06:47:49,901 - 1 day, 13:31:57 - 101.006s - INFO - __main__ - Start training yahoo_answers_csv...
# 2023-06-14 08:34:28,808 - 1 day, 15:18:36 - 6398.907s - INFO - __main__ - progress: 0.13 , step: 500 , lr: 1.91E-05 , avg batch size: 30.0 , avg loss: 3.607
# 2023-06-14 10:22:48,461 - 1 day, 17:06:55 - 6499.653s - INFO - __main__ - progress: 0.27 , step: 1000 , lr: 1.82E-05 , avg batch size: 30.0 , avg loss: 3.598
# 2023-06-14 12:11:11,370 - 1 day, 18:55:18 - 6502.909s - INFO - __main__ - progress: 0.40 , step: 1500 , lr: 1.74E-05 , avg batch size: 30.0 , avg loss: 3.597
# 2023-06-14 14:00:07,420 - 1 day, 20:44:14 - 6536.051s - INFO - __main__ - progress: 0.53 , step: 2000 , lr: 1.65E-05 , avg batch size: 30.0 , avg loss: 3.596
# 2023-06-14 15:49:35,422 - 1 day, 22:33:42 - 6568.001s - INFO - __main__ - progress: 0.66 , step: 2500 , lr: 1.56E-05 , avg batch size: 30.0 , avg loss: 3.594
# 2023-06-14 17:40:30,201 - 2 days, 0:24:37 - 6654.779s - INFO - __main__ - progress: 0.80 , step: 3000 , lr: 1.47E-05 , avg batch size: 30.0 , avg loss: 3.594
# 2023-06-14 19:32:17,087 - 2 days, 2:16:24 - 6706.887s - INFO - __main__ - progress: 0.93 , step: 3500 , lr: 1.39E-05 , avg batch size: 30.0 , avg loss: 3.595
# 2023-06-14 20:28:01,332 - 2 days, 3:12:08 - 3344.245s - INFO - __main__ - Finsih training, avg loss: 3.5952023-06-14 20:28:01,333 - 2 days, 3:12:08 - 0.001s - INFO - __main__ - tot_n_inputs: 115000
# 2023-06-14 20:28:01,334 - 2 days, 3:12:08 - 0.000s - INFO - __main__ - len(train_dataset): 115000
# 2023-06-14 20:28:01,334 - 2 days, 3:12:08 - 0.000s - INFO - __main__ - args.n_train: 115000
# 2023-06-14 20:28:20,160 - 2 days, 3:12:27 - 18.826s - INFO - __main__ - [TIME] End Run at 2023-06-14T20:28:20 within 51.206808956199225 hours



# ./run_v2.sh 0 MetaMBPA_order1_v2
# (1) Change default lr (both inner and outer) 
# (2) Ignore this first. should be able to train without changing this! ---- Adaptation is SGD over 32 x 32 batch, instead of a batch of 32 adapted over 20 times!!!!!
# (3) Write 1%
# (4) Change Loss Calc
# Batch 32 doesn't work with dynamic_collate_fn, need to debug this later!!!
python3 train_v2.py --tasks $TASKS --output_dir "/data/model_runs/em_in_lll/$2" --write_prob 0.01 --inner_lr 1e-5 --learning_rate 3e-5