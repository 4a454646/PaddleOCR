#!/bin/bash
read -p "Enter folder name: " dest

cp inference_results/system_results.txt labeller/system_results/${dest}.txt
