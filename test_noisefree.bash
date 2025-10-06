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
     --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt"   \
     >> test_2_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt"   \
     >> test_2_2.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 2             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt"   \
     >> test_2_5.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k  \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt"   \
     >> test_4_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt"   \
     >> test_4_2.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt"   \
     >> test_4_5.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 4             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt"   \
     >> test_4_1.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt"   \
     >> test_6_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt"   \
     >> test_6_2.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt"   \
     >> test_6_5.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 6             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt"   \
     >> test_6_1.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 8             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/train.csv", "test": "/pvc/valid.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt"   \
     >> test_8_0.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 8             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/2_new_sap/train_noisy.csv", "test": "/pvc/2_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt"   \
     >> test_8_2.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 8             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/5_new_sap/train_noisy.csv", "test": "/pvc/5_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt"   \
     >> test_8_5.txt 2>&1

python /SCRIPTS/test_new.py run \
     --model ags_tiny_unet_50k    \
     --seed 23456             \
     --quant_config 8             \
     --data_path /pvc/            \
     --csv_paths '{"train": "/pvc/1_new_sap/train_noisy.csv", "test": "/pvc/1_new_sap/test_noisy.csv"}'  \
     --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt"   \
     >> test_8_1.txt 2>&1