#!/bin/bash/
export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
read -p "Model: " model

python3 tools/infer/predict_rec.py --image_dir="./aug_train/train_imgs/train" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-train.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/train_imgs/test" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-test.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/train_imgs/cards_crops/front_cards" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-card_crops.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/train_imgs/gc_ori" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/train_imgs-gc_ori.txt
