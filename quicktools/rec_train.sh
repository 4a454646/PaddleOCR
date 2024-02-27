export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/
export FLAGS_use_cuda_managed_memory=true
python3 tools/train.py -c configs/rec_v4.yml