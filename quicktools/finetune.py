import re

def filter_file(input_filename, output_filename):
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        for line in input_file:
            # check if the line is empty; if so, skip it
            if len(line.strip()) == 0:
                continue
            # Split the line into filepath and transcription
            filepath, transcription = line.split('\t')
            
            # strip the transcription of excess spaces and special characters
            transcription = transcription.strip()

            # Check if the transcription contains a space
            has_space = re.search(r'\s', transcription)
            print(f"Transcription: {transcription}, contains space? {has_space}")
            if re.search(r'\s', transcription):
                # If it does, write the line to the output file
                output_file.write(line)

# List of input and output files
files = [
    ('train_data/train_list.txt', 'train_data/train_list_finetune.txt'),
    ('train_data/val_list.txt', 'train_data/val_list_finetune.txt'),
    ('train_data/alt_list.txt', 'train_data/alt_list_finetune.txt'),
]

# Process each file
for input_filename, output_filename in files:
    filter_file(input_filename, output_filename)