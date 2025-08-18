#!/bin/bash

# Save results into ./result_quan_543.txt every 10 minutes
while true; do
    echo "Copying file at $(date)..."
    kubectl cp -n reliable-nn 3090-2-23456-8hshg:2_23456.txt  ./result_quan_2_23456.txt 
    echo "copied for quan bit 2 seed 23456"
    kubectl cp -n reliable-nn 3090-2-34567-72ggk:2_34567.txt  ./result_quan_2_34567.txt 
    echo "copied for quan bit 2 seed 34567"
    kubectl cp -n reliable-nn 3090-4-23456-8zbr6:4_23456.txt  ./result_quan_4_23456.txt 
    echo "copied for quan bit 4 seed 23456"
    kubectl cp -n reliable-nn 3090-4-34567-tzl5t:4_34567.txt  ./result_quan_4_34567.txt 
    echo "copied for quan bit 4 seed 34567"
    kubectl cp -n reliable-nn 3090-6-23456-cvsgz:6_23456.txt  ./result_quan_6_23456.txt 
    echo "copied for quan bit 6 seed 23456"
    kubectl cp -n reliable-nn 3090-6-34567-zltrc:6_34567.txt  ./result_quan_6_34567.txt 
    echo "copied for quan bit 6 seed 34567"
    kubectl cp -n reliable-nn 3090-8-23456-b2p6w:8_23456.txt  ./result_quan_8_23456.txt 
    echo "copied for quan bit 8 seed 23456"
    kubectl cp -n reliable-nn 3090-8-34567-scbxq:8_34567.txt  ./result_quan_8_34567.txt 
    echo "copied for quan bit 8 seed 34567"
    # Wait 10 minutes (600 seconds)
    sleep 800
done