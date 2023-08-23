import re

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

def main():
    # Prompt for input file path
    input_file_path = input("Enter the path of the input text file: ")
    
    # Read input lines from the text file
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    # Extract predictions
    predictions = extract_predictions(lines)

    # Prompt for output file path
    output_file_path = input("Enter the path of the output text file: ")

    # Write the output to the text file
    with open(output_file_path, 'w') as output_file:
        for image_filename, prediction in predictions.items():
            output_file.write(f"{image_filename}\t{prediction}\n")

    print("Extraction completed. Predictions saved to the output text file.")

if __name__ == "__main__":
    main()
