#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH
python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2/train_noisy.csv", "test": "/pvc/2/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_75_test_F1=0.8269.pt" \
    >> result_test_quan_2_noise_2.txt 2>&1

cp result_test_quan_2_noise_2.txt /pvc/result_test_quan_2_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_75_test_F1=0.8269.pt" \
    >> result_test_quan_2_noise_10.txt 2>&1

cp result_test_quan_2_noise_10.txt /pvc/result_test_quan_2_noise_10.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 4 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt" \
    >> result_test_quan_4_noise_10.txt 2>&1

cp result_test_quan_4_noise_10.txt /pvc/result_test_quan_4_noise_10.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 4 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5/train_noisy.csv", "test": "/pvc/5/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt" \
    >> result_test_quan_4_noise_5.txt 2>&1

cp result_test_quan_4_noise_5.txt /pvc/result_test_quan_4_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 4 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2/train_noisy.csv", "test": "/pvc/2/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt" \
    >> result_test_quan_4_noise_2.txt 2>&1

cp result_test_quan_4_noise_2.txt /pvc/result_test_quan_4_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250819-012631/best_model_81_test_F1=0.8416.pt" \
    >> result_test_quan_6_noise_10.txt 2>&1

cp result_test_quan_6_noise_10.txt /pvc/result_test_quan_6_noise_10.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5/train_noisy.csv", "test": "/pvc/5/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250819-012631/best_model_81_test_F1=0.8416.pt" \
    >> result_test_quan_6_noise_5.txt 2>&1

cp result_test_quan_6_noise_5.txt /pvc/result_test_quan_6_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 6 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2/train_noisy.csv", "test": "/pvc/2/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250819-012631/best_model_81_test_F1=0.8416.pt" \
    >> result_test_quan_6_noise_2.txt 2>&1

cp result_test_quan_6_noise_2.txt /pvc/result_test_quan_6_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/10/train_noisy.csv", "test": "/pvc/10/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt" \
    >> result_test_quan_8_noise_10.txt 2>&1

cp result_test_quan_8_noise_10.txt /pvc/result_test_quan_8_noise_10.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/5/train_noisy.csv", "test": "/pvc/5/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt" \
    >> result_test_quan_8_noise_5.txt 2>&1

cp result_test_quan_8_noise_5.txt /pvc/result_test_quan_8_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 8 \
    --data_path /pvc/ \
    --csv_paths '{"train": "/pvc/2/train_noisy.csv", "test": "/pvc/2/test_noisy.csv"}' \
    --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt" \
    >> result_test_quan_8_noise_2.txt 2>&1

cp result_test_quan_8_noise_2.txt /pvc/result_test_quan_8_noise_2.txt
