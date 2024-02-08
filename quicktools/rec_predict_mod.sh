#!/bin/bash/
export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

model="newrec/hardneg_allaugs_lesspower_newset"

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/train_list.txt" \
  --full_paths="./train_data/train_list_paths.txt" \
  --out_path="./trials/${model}/train_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/val_list.txt" \
  --full_paths="./train_data/val_list_paths.txt" \
  --out_path="./trials/${model}/val_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/alt_list.txt" \
  --full_paths="./train_data/alt_list_paths.txt" \
  --out_path="./trials/${model}/alt_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false