import json
import os
import numpy as np
import random

data_dir = 'output'
sub_dirs = []
output_lines = []

for path in os.listdir(data_dir):
    sub_dir = os.path.join(data_dir, path)
    if os.path.isdir(sub_dir):
        sub_dirs.append(sub_dir)

for sub_dir in sub_dirs:
    label_file = os.path.join(sub_dir, 'labels.json')
    with open(label_file) as f:
        data = json.load(f)

    labels = data['labels']

    # Create lines in the desired format
    for key, value in labels.items():
        file_name = sub_dir + '/images/' + key + '.jpg'
        text = value.strip('"')
        output_lines.append(f'{file_name}\t{text}')

# Write the lines to a text file
with open('all_data.txt', 'w') as file:
    file.write('\n'.join(output_lines))