export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/export_model.py \
    -c ./configs/rec/v4_overwritten.yml \
    -o Global.pretrained_model=./trials/alt/new_hardneg/latest \
    Global.save_inference_dir=./trials/alt/new_hardneg/model
