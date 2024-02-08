#!/bin/bash/
export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

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

mkdir -p preds/${model}
touch -a preds/${model}/train_imgs-card_crops.txt
touch -a preds/${model}/train_imgs-gc_ori.txt
touch -a preds/${model}/train_imgs-train.txt
touch -a preds/${model}/train_imgs-test.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/cards_crops/front_cards" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-card_crops.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/gc_ori" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-gc_ori.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/train" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-train.txt


python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/test" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-test.txt



touch -a preds/${model}/train_imgs-train.txt
touch -a preds/${model}/train_imgs-test.txt
touch -a preds/${model}/train_imgs-card_crops.txt
touch -a preds/${model}/train_imgs-gc_ori.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/train" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-train.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/test" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-test.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/cards_crops/front_cards" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-card_crops.txt

python3 tools/infer/predict_rec.py --image_dir="./train_data/train_imgs/gc_ori" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-gc_ori.txt
