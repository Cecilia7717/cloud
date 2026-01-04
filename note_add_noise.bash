#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH

python /SCRIPTS/train_noisy.py run --model ags_tiny_unet_50k --seed 23456 --quant_config 8 --data_path /pvc/ --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}' --noise_module 2.0 >> make_noisy_2.txt 2>&1 
python /SCRIPTS/train_noisy.py run --model ags_tiny_unet_50k --seed 23456 --quant_config 8 --data_path /pvc/ --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}' --noise_module 1.0 >> make_noisy_1.txt 2>&1 
python /SCRIPTS/train_noisy.py run --model ags_tiny_unet_50k --seed 23456 --quant_config 8 --data_path /pvc/ --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}' --noise_module 5.0 >> make_noisy_5.txt 2>&1 
