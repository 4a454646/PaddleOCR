#!/bin/bash/
export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

while getopts "m:" opt; do
  case $opt in
    m)
      model=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

python3 tools/export_model.py \
    -c ./trials/cosine_0.1_aug/config.yml \
    -o Global.pretrained_model=./trials/${model}/latest \
    Global.save_inference_dir=./trials/${model}/model

model="${model}/model"

python3 -u tools/infer/predict_rec_mod.py \
  --label="train" \
  --train_list="./aug_train/train_list.txt" \
  --full_paths="./aug_train/train_list_paths.txt" \
  --out_path="./trials/${model}/train_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 -u tools/infer/predict_rec_mod.py \
  --label="validation" \
  --train_list="./aug_train/val_list.txt" \
  --full_paths="./aug_train/val_list_paths.txt" \
  --out_path="./trials/${model}/val_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false

python3 -u tools/infer/predict_rec_mod.py \
  --label="test" \
  --train_list="./aug_train/test_list.txt" \
  --full_paths="./aug_train/test_list_paths.txt" \
  --out_path="./trials/${model}/test_result.txt" \
  --rec_model_dir="./trials/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false