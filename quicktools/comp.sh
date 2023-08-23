#!/bin/bash/
read -p "Enter folder name: " dest

wc pred_labels/${dest}_preds.txt 

find rec_crops/${dest}_crops -type f | wc -l
