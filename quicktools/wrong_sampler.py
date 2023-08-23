def compare_and_write(file1_path, file2_path, output_file_path):
    with open(file1_path, 'r') as file1:
        lines1 = file1.readlines()

    with open(file2_path, 'r') as file2:
        lines2 = file2.readlines()

    # Find dissimilar lines
    diff_lines = [line1 for line1, line2 in zip(lines1, lines2) if line1.split('\t')[-1] != line2.split('\t')[-1]]

    # Write dissimilar lines to the output file
    with open(output_file_path, 'w') as output_file:
        for line in diff_lines:
            output_file.write(line)

    print("Comparison completed. Dissimilar lines written to the output file.")

def main():
    # Prompt for file paths
    file1_path = input("Enter the path of the first text file: ")
    file2_path = input("Enter the path of the second text file: ")
    output_file_path = input("Enter the path of the output text file: ")

    # Compare and write dissimilar lines
    compare_and_write(file1_path, file2_path, output_file_path)

if __name__ == "__main__":
    main()
