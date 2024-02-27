export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/
export FLAGS_use_cuda_managed_memory=true
python3 tools/hyperparam.py -c configs/rec/v4_overwritten.yml