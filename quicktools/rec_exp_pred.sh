#!/bin/bash/
export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/

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
    -c ./rec_train/${model}/config.yml \
    -o Global.pretrained_model=./rec_train/${model}/latest \
    Global.save_inference_dir=./rec_train/${model}/model

model="${model}/model"

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/train_list.txt" \
  --full_paths="./train_data/train_list_paths.txt" \
  --out_path="./rec_train/${model}/train_result.txt" \
  --rec_model_dir="./rec_train/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false \
  --label="Train"

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/val_list.txt" \
  --full_paths="./train_data/val_list_paths.txt" \
  --out_path="./rec_train/${model}/val_result.txt" \
  --rec_model_dir="./rec_train/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false \ 
  --label="Val"

python3 tools/infer/predict_rec_mod.py \
  --train_list="./train_data/alt_list.txt" \
  --full_paths="./train_data/alt_list_paths.txt" \
  --out_path="./rec_train/${model}/alt_result.txt" \
  --rec_model_dir="./rec_train/${model}/" \
  --rec_image_shape="3, 48, 320" \
  --rec_char_dict_path="./ppocr/utils/en_dict.txt" \
  --show_log=false \
  --label="Alt"