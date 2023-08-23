#!/bin/bash/

read -p "Model: " model

python3 tools/infer/predict_rec.py --image_dir="./aug_train/val_imgs/gc_ori" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/val_gc.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/val_imgs/test" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/val_test.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/val_imgs/cards_crops/headings" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/val_cards_crops.txt

python3 tools/infer/predict_rec.py --image_dir="./aug_train/val_imgs/rec_test_crops" --rec_model_dir="./inference/${model}/" --rec_image_shape="3, 48, 320" --rec_char_dict_path="ppocr/utils/en_dict.txt" > preds/${model}/val_rec_test.txt
