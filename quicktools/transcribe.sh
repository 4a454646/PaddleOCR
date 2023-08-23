#!/bin/bash
read -p "Enter folder name: " dest

python3 tools/infer/predict_system.py --image_dir="./labeller/src_images/${dest}" --det_model_dir="./inference/base_det/"  --rec_model_dir="./inference/base_rec/" --rec_char_dict_path="ppocr/utils/en_dict.txt"
