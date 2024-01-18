export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

model_path=$1

python3 tools/export_model.py \
    -c "./trials/${model_path}/config.yml" \
    -o Global.pretrained_model="./trials/${model_path}/latest" \
    Global.save_inference_dir="./trials/${model_path}/model"