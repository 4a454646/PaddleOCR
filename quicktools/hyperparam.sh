export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
export FLAGS_use_cuda_managed_memory=true
python3 tools/hyperparam.py -c configs/rec/v4_overwritten.yml