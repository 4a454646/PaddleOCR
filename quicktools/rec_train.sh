export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy
