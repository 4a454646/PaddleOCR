import re
import os
import argparse

def extract_predictions(lines):
    pattern = r"Predicts of .*?/([^/]+/[^/]+\.(?:jpg|png)):\(['\"](.+?)['\"]\s*,"
    extracted_predictions = {}

    for line in lines:
        match = re.search(pattern, line)
        if match:
            image_filename = match.group(1)
            prediction = match.group(2)
            extracted_predictions[image_filename] = prediction

    return extracted_predictions

def process_file(input_file_path, output_file_path):
    # Read input lines from the text file
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    # Extract predictions
    predictions = extract_predictions(lines)

    # Write the output to the text file
    with open(output_file_path, 'w') as output_file:
        for image_filename, prediction in predictions.items():
            output_file.write(f"{image_filename}\t{prediction}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_path', help='Path to the folder containing input text files')
    args = parser.parse_args()
    full_path = f"preds/{args.folder_path}"

    for filename in os.listdir(full_path):
        if filename.endswith('.txt') and not filename.endswith('-export.txt'):
            input_file_path = os.path.join(full_path, filename)
            output_file_path = os.path.join(full_path, f"{os.path.splitext(filename)[0]}-export.txt")
            process_file(input_file_path, output_file_path)