export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/export_model.py -c ./trials/cosine_0.1_aug/config.yml -o Global.pretrained_model=./trials/cosine_imgaug_25_rotation/latest Global.save_inference_dir=./inference/cosine_imgaug_25_rotation
