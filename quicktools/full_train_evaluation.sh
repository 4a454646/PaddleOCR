#!/bin/bash

# Define the output file name
output_file="results.txt"

# Loop through each folder in /workspace/PaddleOCR/trials/v4
for folder in /workspace/PaddleOCR/trials/v4/*; do
  # Get the folder name
  folder_name=$(basename "$folder")
  
  # Run the quicktools/exp_and_pred.sh script with the -m flag set to the folder name
  bash quicktools/exp_and_pred.sh -m "v4/$folder_name" > "$folder_name.txt"
  
  # Append the output to the main output file with the folder name as a label
  echo "Results for $folder_name:" >> "$output_file"
  cat "$folder_name.txt" >> "$output_file"
  echo "" >> "$output_file"
  
  # Remove the temporary output file
  rm "$folder_name.txt"
done