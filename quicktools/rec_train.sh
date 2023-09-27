export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
export FLAGS_use_cuda_managed_memory=true
python3 tools/train.py -c configs/rec/v4_overwritten.yml