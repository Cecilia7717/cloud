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
    --csv_paths "/pvc/2/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt" \
    >> result_test_quan_2_noise_2.txt 2>&1

cp result_test_quan_2_noise_2.txt /pvc/result_test_quan_2_noise_2.txt

python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/5/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt" \
    >> result_test_quan_2_noise_5.txt 2>&1

cp result_test_quan_2_noise_5.txt /pvc/result_test_quan_2_noise_5.txt


python /SCRIPTS/test.py test-only \
    --model ags_tiny_unet_50k \
    --seed 23456 \
    --quant_config 2 \
    --data_path /pvc/ \
    --csv_paths "/pvc/10/test_noisy.csv" \
    --resume_from="/pvc/output-alcd-cloud-quan-2-23456/ags_tiny_unet_50k_backend-None-1_20250818-182321/best_model_78_test_F1=0.8281.pt" \
    >> result_test_quan_2_noise_10.txt 2>&1

cp result_test_quan_2_noise_10.txt /pvc/result_test_quan_2_noise_10.txt
