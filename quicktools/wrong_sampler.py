import argparse
import os

def compare_and_write(file1_path, file2_path, output_file_path):
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()

    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    # Find dissimilar lines
    diff_lines = [(line1, line2) for line1, line2 in zip(lines1, lines2) if line1.split('\t')[-1] != line2.split('\t')[-1]]

    # Write dissimilar lines to the output file
    with open(output_file_path, 'w') as output_file:
        for i in range(len(diff_lines)):
            line = diff_lines[i][0]
            output = line.rstrip() + "\t (expected: {})".format(diff_lines[i][1].split('\t')[-1].rstrip()) + "\n"
            output_file.write(output)

    print("Comparison completed. Dissimilar lines written to the output file.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', help='Path to the model directory')
    parser.add_argument('-i', '--input_file1', help='Path to the first input file')
    parser.add_argument('-ii', '--input_file2', help='Path to the second input file')
    args = parser.parse_args()

    # Compare the input files and write the output file
    fullpath_1 = f"preds/{args.model_dir}/{args.input_file1}"
    fullpath_2 = f"aug_train/{args.input_file2}"
    os.makedirs(f"preds/{args.model_dir}/diffs", exist_ok=True)
    output_file_path = f"preds/{args.model_dir}/diffs/{args.input_file1}"
    compare_and_write(fullpath_1, fullpath_2, output_file_path)
