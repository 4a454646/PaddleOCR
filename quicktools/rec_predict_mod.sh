#!/bin/bash/
export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

model="v4/0.4_bda/model"

python3 tools/infer/predict_rec_mod.py \
  --train_list="./aug_train/train_list.txt" \
  --full_paths="./aug_train/train_list_paths.txt" \
  --out_path="./trials/${model}/train_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 tools/infer/predict_rec_mod.py \
  --train_list="./aug_train/val_list.txt" \
  --full_paths="./aug_train/val_list_paths.txt" \
  --out_path="./trials/${model}/val_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 tools/infer/predict_rec_mod.py \
  --train_list="./aug_train/test_list.txt" \
  --full_paths="./aug_train/test_list_paths.txt" \
  --out_path="./trials/${model}/test_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false