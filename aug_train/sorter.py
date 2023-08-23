def sort_by_first_block(file_path):
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    sorted_lines = sorted(lines, key=lambda line: line.split('\t')[0])
    with open(file_path, 'w') as outfile:
        outfile.writelines(sorted_lines)

# Example usage:
file_path = input("Your text file: ")
sort_by_first_block(file_path)

