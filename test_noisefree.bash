#!/bin/bash

# set master addr/port (change port if needed)
export MASTER_ADDR=localhost
export MASTER_PORT=29600
export PYTHONPATH=/root:$PYTHONPATH
# for model trained with quan 2 noise 2
python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/2_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt" \
    >> result_test_quan_2_noise_2.txt 2>&1

cp result_test_quan_2_noise_2.txt /pvc/result_test_quan_2_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/5_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt" \
    >> result_test_quan_2_noise_5.txt 2>&1

cp result_test_quan_2_noise_5.txt /pvc/result_test_quan_2_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/2_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt" \
    >> result_test_quan_4_noise_2.txt 2>&1

cp result_test_quan_4_noise_2.txt /pvc/result_test_quan_4_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/5_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-4-23456/ags_tiny_unet_50k_backend-None-1_20250818-182859/best_model_84_test_F1=0.8402.pt" \
    >> result_test_quan_4_noise_5.txt 2>&1

cp result_test_quan_4_noise_5.txt /pvc/result_test_quan_4_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/2_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt" \
    >> result_test_quan_6_noise_2.txt 2>&1

cp result_test_quan_6_noise_2.txt /pvc/result_test_quan_6_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/5_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-6-23456/ags_tiny_unet_50k_backend-None-1_20250818-183500/best_model_51_test_F1=0.8402.pt" \
    >> result_test_quan_6_noise_5.txt 2>&1

cp result_test_quan_6_noise_5.txt /pvc/result_test_quan_6_noise_5.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/2_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt" \
    >> result_test_quan_8_noise_2.txt 2>&1

cp result_test_quan_8_noise_2.txt /pvc/result_test_quan_8_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/5_new/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-8-23456/ags_tiny_unet_50k_backend-None-1_20250818-183924/best_model_114_test_F1=0.8516.pt" \
    >> result_test_quan_8_noise_5.txt 2>&1

cp result_test_quan_8_noise_5.txt /pvc/result_test_quan_8_noise_5.txt