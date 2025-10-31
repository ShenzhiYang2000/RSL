#!/bin/bash

python main.py --dataset amazon --selected_number 2 --lr 0.005 --extra 1.0 --alpha 0.5 --sgld_lr 0.0 --sgld_std 0.0 --detect_times 17
python main.py --dataset yelp --selected_number 2 --lr 0.005 --extra 1.0 --alpha 0.5 --sgld_lr 0.0 --sgld_std 0.0 --detect_times 58
python main.py --dataset reddit --selected_number 1 --lr 0.01 --extra 1.0 --alpha 0.5 --sgld_lr 0.0 --sgld_std 0.0 --detect_times 16
python main.py --dataset wikics --selected_number 1 --lr 0.01 --extra 1.0 --alpha 0.5 --detect_times 47
python main.py --dataset squirrel --selected_number 2 --lr 0.005 --patience 20 --w_decay 0.001  --extra 0.001 --alpha 1.0 --detect_times 84



