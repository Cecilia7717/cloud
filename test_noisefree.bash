#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/train.csv" \
     --csv_paths.test "/pvc/valid.csv" \
     >> test_8_0.txt 2>&1

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/2_new_sap/train_noisy.csv" \
     --csv_paths.test "/pvc/2_new_sap/test_noisy.csv" \
     >> test_8_2_sap.txt 2>&1

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/2_new/train_noisy.csv" \
     --csv_paths.test "/pvc/2_new/test_noisy.csv" \
     >> test_8_2_gn.txt 2>&1

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/5_new_sap/train_noisy.csv" \
     --csv_paths.test "/pvc/5_new_sap/test_noisy.csv" \
     >> test_8_5_sap.txt 2>&1

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/5_new/train_noisy.csv" \
     --csv_paths.test "/pvc/5_new/test_noisy.csv" \
     >> test_8_5_gn.txt 2>&1

python SCRIPTS/test_normal.py eval \
     --model ags_tiny_unet_50k \
     --seed 23456             \
     --quant_config 8             \
     --resume_from "/pvc/output-8/ags_tiny_unet_50k_backend-None-1_20251103-062633/best_checkpoint_78_test_F1=0.8466.pt" \
     --data_path /pvc/ \
     --csv_paths.train "/pvc/1_new_sap/train_noisy.csv" \
     --csv_paths.test "/pvc/1_new_sap/test_noisy.csv" \
     >> test_8_1_sap.txt 2>&1
