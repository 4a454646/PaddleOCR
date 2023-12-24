export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/export_model.py \
    -c ./configs/rec/v4_overwritten.yml \
    -o Global.pretrained_model=./trials/newrec/horiz_withpretrain/latest \
    Global.save_inference_dir=./trials/newrec/horiz_withpretrain/model
