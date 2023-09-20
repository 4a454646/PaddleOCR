with open('/workspace/PaddleOCR/aug_train/test_list.txt', 'r') as input_file:
    with open('/workspace/PaddleOCR/aug_train/test_list_paths.txt', 'w') as output_file:
        for line in input_file:
            file_path = line.split('\t')[0]
            output_file.write(f"/workspace/PaddleOCR/aug_train/{file_path}\n")