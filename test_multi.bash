#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH
# for model trained with quan 2 noise 2
python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 4 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2_new/train_noisy.csv", "test": "/pvc/2_new/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-4/ags_tiny_unet_50k_backend-None-1_20250909-002426/best_model_93_test_F1=0.8390.pt" \
    >> result_quan_4_gn_10_test_noise_2.txt 2>&1

cp result_quan_4_gn_10_test_noise_2.txt /pvc/result_quan_4_gn_10_test_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 4 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5_new/train_noisy.csv", "test": "/pvc/5_new/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-4/ags_tiny_unet_50k_backend-None-1_20250909-002426/best_model_93_test_F1=0.8390.pt" \
    >> result_quan_4_gn_10_test_noise_5.txt 2>&1

cp result_quan_4_gn_10_test_noise_5.txt /pvc/result_quan_4_gn_10_test_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2_new/train_noisy.csv", "test": "/pvc/2_new/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-6/ags_tiny_unet_50k_backend-None-1_20250909-112342/best_model_93_test_F1=0.8392.pt" \
    >> result_quan_6_gn_10_test_noise_2.txt 2>&1

cp result_quan_6_gn_10_test_noise_2.txt /pvc/result_quan_6_gn_10_test_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5/train_noisy.csv", "test": "/pvc/5/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-6/ags_tiny_unet_50k_backend-None-1_20250909-112342/best_model_93_test_F1=0.8392.pt" \
    >> result_quan_6_gn_10_test_noise_5.txt 2>&1

cp result_quan_6_gn_10_test_noise_5.txt /pvc/result_quan_6_gn_10_test_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-6/ags_tiny_unet_50k_backend-None-1_20250909-112342/best_model_93_test_F1=0.8392.pt" \
    >> result_quan_6_gn_10_test_noise_10.txt 2>&1

cp result_quan_6_gn_10_test_noise_10.txt /pvc/result_quan_6_gn_10_test_noise_10.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2/train_noisy.csv", "test": "/pvc/2/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-8/ags_tiny_unet_50k_backend-None-1_20250909-023033/best_model_90_test_F1=0.8408.pt" \
    >> result_quan_8_gn_10_test_noise_2.txt 2>&1

cp result_quan_8_gn_10_test_noise_2.txt /pvc/result_quan_8_gn_10_test_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5/train_noisy.csv", "test": "/pvc/5/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-8/ags_tiny_unet_50k_backend-None-1_20250909-023033/best_model_90_test_F1=0.8408.pt" \
    >> result_quan_8_gn_10_test_noise_5.txt 2>&1

cp result_quan_8_gn_10_test_noise_5.txt /pvc/result_quan_8_gn_10_test_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-noisy-10-quan-8/ags_tiny_unet_50k_backend-None-1_20250909-023033/best_model_90_test_F1=0.8408.pt" \
    >> result_quan_8_gn_10_test_noise_10.txt 2>&1

cp result_quan_8_gn_10_test_noise_10.txt /pvc/result_quan_8_gn_10_test_noise_10.txt