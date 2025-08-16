#!/bin/bash

# Save results into ./result_quan_543.txt every 10 minutes
while true; do
    echo "Copying file at $(date)..."
    kubectl cp -n reliable-nn inter3090-6:5_27.txt  ./result_quan_527_6.txt 
    echo "copied for quan bit 6"
    kubectl cp -n reliable-nn inter3090-2:5_00.txt  ./result_quan_500_2.txt
    echo "copied for quan bit 2"
    # Wait 10 minutes (600 seconds)
    sleep 800
done
