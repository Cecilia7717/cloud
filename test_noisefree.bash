#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_2_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new/train_noisy.csv", "test": "/pvc/2_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_2_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_5_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new/train_noisy.csv", "test": "/pvc/5_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_5_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-2-new-saveall/best_checkpoint_93_test_F1=0.8279.pt"   \
     >> test_2_1_sap.txt 2>&1


python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_2_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new/train_noisy.csv", "test": "/pvc/2_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_2_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_5_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new/train_noisy.csv", "test": "/pvc/5_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_5_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-4-new-saveall/best_checkpoint_93_test_F1=0.8458.pt"   \
     >> test_4_1_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_2_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new/train_noisy.csv", "test": "/pvc/2_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_2_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_5_sap.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new/train_noisy.csv", "test": "/pvc/5_new/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_5_gn.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-6-new-saveall/best_checkpoint_81_test_F1=0.8455.pt"   \
     >> test_6_1_sap.txt 2>&1
