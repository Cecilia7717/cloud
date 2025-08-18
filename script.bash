#!/bin/bash

# Save results into ./result_quan_543.txt every 10 minutes
while true; do
    echo "Copying file at $(date)..."
    kubectl cp -n reliable-nn inter3090-72q5n:6_25.txt  ./result_quan_625_6.txt 
    echo "copied for quan bit 6"
    kubectl cp -n reliable-nn inter3090-2-64r8h:9_06.txt  ./result_quan_906_2.txt 
    echo "copied for quan bit 2"
    # Wait 10 minutes (600 seconds)
    sleep 800
done