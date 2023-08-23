import os
import shutil

def process_text_file(file_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the line at the tab character
                parts = line.split('\t')
                if len(parts) > 0:
                    image_path = parts[0]
                    # Get the filename from the image path
                    # image_name = os.path.join('output', image_path)
                    # print(image_name)
                    # Create the output path for the image
                    # img_name = image_path.replace("/", "@")
                    output_path = os.path.join(output_folder, image_path)
                    # print(output_path)
                    # Create the intermediate directories if they don't exist
                    destination_folder = os.path.dirname(output_path)
                    os.makedirs(destination_folder, exist_ok=True)
                    # Copy or move the image to the output folder
                    shutil.copy(image_path, output_path)
                    # If you want to move the image instead, replace shutil.copy with shutil.move
                    print(f"Processed: {image_path}")

# Provide the path to your input text file
input_file_path = 'test_set.txt'

# Provide the path to the output folder where the images will be stored
output_folder = 'rup_test'

process_text_file(input_file_path, output_folder)
